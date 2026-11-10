// gow3: SDL3 window for the Vulkan swapchain (X11 or Wayland).
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <SDL3/SDL.h>
#include "common/assert.h"
#include "common/logging/log.h"
#include "sdl_window.h"
#include "gow3_overlay.h"
#include "gow3_graphics.h"
#include "gow3_settings.h"

namespace Frontend {

WindowSDL::WindowSDL(s32 width_, s32 height_, const char* title) : width{width_}, height{height_} {
    // Gamepads are sampled by runtime_pad.c; their events are pumped here with the window's.
    // They keep working while another window has the focus (the launcher, a second screen):
    // without this a pad stayed silent until the game window was clicked (issue #25).
    SDL_SetHint(SDL_HINT_JOYSTICK_ALLOW_BACKGROUND_EVENTS, "1");
    if (!SDL_InitSubSystem(SDL_INIT_VIDEO | SDL_INIT_GAMEPAD)) {
        UNREACHABLE_MSG("Failed to initialize SDL video: {}", SDL_GetError());
    }
    SDL_PropertiesID props = SDL_CreateProperties();
    SDL_SetStringProperty(props, SDL_PROP_WINDOW_CREATE_TITLE_STRING, title);
    SDL_SetNumberProperty(props, SDL_PROP_WINDOW_CREATE_X_NUMBER, SDL_WINDOWPOS_CENTERED);
    SDL_SetNumberProperty(props, SDL_PROP_WINDOW_CREATE_Y_NUMBER, SDL_WINDOWPOS_CENTERED);
    SDL_SetNumberProperty(props, SDL_PROP_WINDOW_CREATE_WIDTH_NUMBER, width_);
    SDL_SetNumberProperty(props, SDL_PROP_WINDOW_CREATE_HEIGHT_NUMBER, height_);
    SDL_SetBooleanProperty(props, SDL_PROP_WINDOW_CREATE_RESIZABLE_BOOLEAN, true);
    SDL_SetBooleanProperty(props, SDL_PROP_WINDOW_CREATE_VULKAN_BOOLEAN, true);
    base_title = title;
    window = SDL_CreateWindowWithProperties(props);
    SDL_DestroyProperties(props);
    ASSERT_MSG(window, "Failed to create window: {}", SDL_GetError());
    // gow3: the game's own icon (run.py converts sce_sys/icon0.png to BMP, SDL's built-in format).
    if (const char* icon_path = std::getenv("GOW3_WINDOW_ICON")) {
        if (SDL_Surface* icon = SDL_LoadBMP(icon_path)) {
            SDL_SetWindowIcon(window, icon);
            SDL_DestroySurface(icon);
        }
    }

    const char* driver = SDL_GetCurrentVideoDriver();
    const SDL_PropertiesID wp = SDL_GetWindowProperties(window);
#ifdef _WIN32
    if (driver && !std::strcmp(driver, "windows")) {
        window_info.type = WindowSystemType::Windows;
        window_info.render_surface = SDL_GetPointerProperty(wp, SDL_PROP_WINDOW_WIN32_HWND_POINTER, nullptr);
    } else
#endif
    if (driver && !std::strcmp(driver, "x11")) {
        window_info.type = WindowSystemType::X11;
        window_info.display_connection = SDL_GetPointerProperty(wp, SDL_PROP_WINDOW_X11_DISPLAY_POINTER, nullptr);
        window_info.render_surface = reinterpret_cast<void*>(SDL_GetNumberProperty(wp, SDL_PROP_WINDOW_X11_WINDOW_NUMBER, 0));
    } else if (driver && !std::strcmp(driver, "wayland")) {
        window_info.type = WindowSystemType::Wayland;
        window_info.display_connection = SDL_GetPointerProperty(wp, SDL_PROP_WINDOW_WAYLAND_DISPLAY_POINTER, nullptr);
        window_info.render_surface = SDL_GetPointerProperty(wp, SDL_PROP_WINDOW_WAYLAND_SURFACE_POINTER, nullptr);
    } else {
        UNREACHABLE_MSG("Unsupported SDL video driver {}", driver ? driver : "(none)");
    }
    // gow3: GOW3_FULLSCREEN=1 (older launchers) still asks for fullscreen.
    if (const char* fullscreen = std::getenv("GOW3_FULLSCREEN"); fullscreen && fullscreen[0] == '1' &&
        Gow3Settings::Get().display_mode == Gow3Graphics::Windowed) {
        Gow3Settings::Get().display_mode = Gow3Graphics::Borderless;
    }
    FitWindowedSize(width_, height_);
    ApplyDisplayMode(Gow3Settings::Get().display_mode);
    int w = 0, h = 0;
    SDL_GetWindowSizeInPixels(window, &w, &h);
    width = w;
    height = h;
    LOG_INFO(Frontend, "Window {}x{} on {}", w, h, driver);
}

// gow3: the windowed size in pixels is the game's output (VideoOut) size, so a 1440p game shows 1:1
// in its window. SDL sizes windows in points: with Windows display scaling a point is several
// pixels. A window that does not fit the desktop's work area (with its frame) keeps the aspect
// ratio at the largest size that fits.
void WindowSDL::FitWindowedSize(s32 pixel_width, s32 pixel_height) {
    int points_w = 0, points_h = 0, pixels_w = 0, pixels_h = 0;
    SDL_GetWindowSize(window, &points_w, &points_h);
    SDL_GetWindowSizeInPixels(window, &pixels_w, &pixels_h);
    const double density = points_w > 0 && pixels_w > 0 ? double(pixels_w) / points_w : 1.0;
    double w = pixel_width / density, h = pixel_height / density;
    SDL_Rect usable{};
    int top = 0, left = 0, bottom = 0, right = 0;
    SDL_GetWindowBordersSize(window, &top, &left, &bottom, &right);
    if (SDL_GetDisplayUsableBounds(SDL_GetDisplayForWindow(window), &usable)) {
        const double fit = std::min({1.0, (usable.w - left - right) / w, (usable.h - top - bottom) / h});
        w *= fit;
        h *= fit;
    }
    SDL_SetWindowSize(window, int(w + 0.5), int(h + 0.5));
    SDL_SetWindowPosition(window, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED);
    SDL_SyncWindow(window);
    SDL_GetWindowSizeInPixels(window, &pixels_w, &pixels_h);
    std::printf("Window: windowed size %dx%d pixels for a %dx%d game image\n", pixels_w, pixels_h, pixel_width,
                pixel_height);
}

// Windowed, borderless (desktop-sized, no mode change) or exclusive fullscreen at the desktop
// mode. Window thread only (SDL). The swapchain follows the size events.
void WindowSDL::ApplyDisplayMode(int mode) {
    display_mode = mode;
    if (mode == Gow3Graphics::Windowed) {
        SDL_SetWindowFullscreen(window, false);
    } else {
        const SDL_DisplayMode* desktop = SDL_GetDesktopDisplayMode(SDL_GetDisplayForWindow(window));
        const bool exclusive = mode == Gow3Graphics::Fullscreen && desktop;
        if (!SDL_SetWindowFullscreenMode(window, exclusive ? desktop : nullptr)) {
            LOG_WARNING(Frontend, "Fullscreen mode: {}", SDL_GetError());
        }
        SDL_SetWindowFullscreen(window, true);
    }
    SDL_SyncWindow(window);
    std::printf("Window: %s\n", Gow3Graphics::DisplayModeNames[mode]);
}

WindowSDL::~WindowSDL() {
    SDL_DestroyWindow(window);
}

void WindowSDL::BeginTextInput(const std::string& initial, const std::string& prompt) {
    std::scoped_lock lock{text_mutex};
    text = initial;
    text_prompt = prompt;
    text_state = 0;
    text_requested = true;
}

int WindowSDL::PollTextInput(std::string& out) {
    std::scoped_lock lock{text_mutex};
    out = text;
    return text_state;
}

void WindowSDL::UpdateTextTitle() {
    const std::string title = text_active ? base_title + " \u2014 " + text_prompt + ": " + text + "_  (Enter = OK, Esc = cancel)"
                                          : base_title;
    SDL_SetWindowTitle(window, title.c_str());
    Gow3Overlay::SetTextEntry(text_active, text_prompt, text);
}

bool WindowSDL::PollEvents() {
    {
        std::scoped_lock lock{text_mutex};
        if (text_requested) { // SDL text input must be toggled from the window thread
            text_requested = false;
            text_active = true;
            SDL_StartTextInput(window);
            UpdateTextTitle();
        }
    }
    if (!text_active) {
        Gow3Overlay::UpdateTextInput(window);
    }
    if (const int mode = Gow3Settings::Get().display_mode; mode != display_mode) {
        ApplyDisplayMode(mode);
    }
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
        if (text_active && (event.type == SDL_EVENT_TEXT_INPUT || event.type == SDL_EVENT_KEY_DOWN)) {
            std::scoped_lock lock{text_mutex};
            if (event.type == SDL_EVENT_TEXT_INPUT) {
                text += event.text.text;
            } else if (event.key.key == SDLK_BACKSPACE && !text.empty()) {
                size_t cut = text.size() - 1; // drop one UTF-8 code point
                while (cut > 0 && (static_cast<unsigned char>(text[cut]) & 0xC0) == 0x80) --cut;
                text.erase(cut);
            } else if (event.key.key == SDLK_RETURN || event.key.key == SDLK_KP_ENTER || event.key.key == SDLK_ESCAPE) {
                text_state = event.key.key == SDLK_ESCAPE ? 2 : 1;
                text_active = false;
                SDL_StopTextInput(window);
            }
            UpdateTextTitle();
            continue;
        }
        // The controller finishes the text dialog too: Cross (A) accepts, Circle (B) cancels.
        if (text_active && event.type == SDL_EVENT_GAMEPAD_BUTTON_DOWN &&
            (event.gbutton.button == SDL_GAMEPAD_BUTTON_SOUTH ||
             event.gbutton.button == SDL_GAMEPAD_BUTTON_EAST)) {
            std::scoped_lock lock{text_mutex};
            text_state = event.gbutton.button == SDL_GAMEPAD_BUTTON_SOUTH ? 1 : 2;
            text_active = false;
            SDL_StopTextInput(window);
            UpdateTextTitle();
            continue;
        }
        if (event.type == SDL_EVENT_MOUSE_MOTION) {
            last_mouse_motion_ms = SDL_GetTicks();
        }
        if (Gow3Overlay::HandleEvent(event)) {
            continue;
        }
        switch (event.type) {
        case SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED:
        case SDL_EVENT_WINDOW_RESIZED: {
            int w = 0, h = 0;
            SDL_GetWindowSizeInPixels(window, &w, &h);
            width = w;
            height = h;
            break;
        }
        case SDL_EVENT_QUIT:
        case SDL_EVENT_WINDOW_CLOSE_REQUESTED:
            is_open = false;
            break;
        default:
            break;
        }
    }
    UpdateCursor();
    return is_open;
}

// The OS cursor over the game (from the Bloodborne port, 4d4d208): hidden in borderless and
// fullscreen, and in a window after 3 s without moving the mouse; always shown while the menu,
// the save list or a text entry is open. Without input focus (alt-tab, another window clicked)
// the cursor is left shown; it hides again once the player is back in the game window.
void WindowSDL::UpdateCursor() {
    const SDL_WindowFlags flags = SDL_GetWindowFlags(window);
    const bool focused = (flags & SDL_WINDOW_INPUT_FOCUS) != 0;
    const bool fullscreen = (flags & SDL_WINDOW_FULLSCREEN) != 0;
    const bool hide = focused && !Gow3Overlay::CapturesInput() &&
                      (fullscreen || SDL_GetTicks() - last_mouse_motion_ms > 3000);
    if (hide != cursor_hidden) {
        cursor_hidden = hide;
        hide ? SDL_HideCursor() : SDL_ShowCursor();
    }
}

} // namespace Frontend
