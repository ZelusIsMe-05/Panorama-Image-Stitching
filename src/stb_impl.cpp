// ============================================================================
// stb_impl.cpp - stb implementation unit + image_io.h functions
// ============================================================================

// stb is third-party code used unmodified: silence its harmless warnings only.
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#endif
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#include "image_io.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <iostream>

bool load_image_rgb(const std::string& path, Image& img) {
    int width = 0, height = 0, channels_in_file = 0;
    unsigned char* pixels = stbi_load(path.c_str(), &width, &height, &channels_in_file, 3);
    if (!pixels) {
        std::cerr << "[I/O Error] Cannot load '" << path << "': " << stbi_failure_reason() << "\n";
        return false;
    }
    img = Image(width, height, 3);
    std::memcpy(img.data.data(), pixels, img.data.size());
    stbi_image_free(pixels);
    return true;
}

bool save_image(const std::string& path, const Image& img) {
    if (!img.is_valid()) {
        std::cerr << "[I/O Error] Nothing to save to '" << path << "'.\n";
        return false;
    }
    std::string ext;
    const size_t dot = path.find_last_of('.');
    if (dot != std::string::npos) ext = path.substr(dot + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

    int ok = 0;
    if (ext == "jpg" || ext == "jpeg")
        ok = stbi_write_jpg(path.c_str(), img.width, img.height, img.channels, img.data.data(), 95);
    else if (ext == "bmp")
        ok = stbi_write_bmp(path.c_str(), img.width, img.height, img.channels, img.data.data());
    else
        ok = stbi_write_png(path.c_str(), img.width, img.height, img.channels, img.data.data(),
                            img.width * img.channels);
    if (!ok) std::cerr << "[I/O Error] Cannot write '" << path << "'.\n";
    return ok != 0;
}
