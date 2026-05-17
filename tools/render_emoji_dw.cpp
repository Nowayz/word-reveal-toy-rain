#define UNICODE
#define _UNICODE

#include <windows.h>
#include <d2d1_3.h>
#include <d3d11.h>
#include <dwrite_3.h>
#include <dxgi1_2.h>
#include <wincodec.h>
#include <stdio.h>
#include <stdlib.h>

#pragma comment(lib, "d2d1.lib")
#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dwrite.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "ole32.lib")
#pragma comment(lib, "windowscodecs.lib")

#define SAFE_RELEASE(x) do { if ((x) != nullptr) { (x)->Release(); (x) = nullptr; } } while (0)

static void check_hr(const wchar_t *step, HRESULT hr) {
    if (FAILED(hr)) {
        fwprintf(stderr, L"%ls failed: 0x%08lx\n", step, (unsigned long)hr);
        ExitProcess(1);
    }
}

static UINT parse_uint_or_default(const wchar_t *value, UINT fallback) {
    if (value == nullptr || *value == L'\0') return fallback;
    wchar_t *end = nullptr;
    unsigned long parsed = wcstoul(value, &end, 10);
    if (end == value || parsed == 0 || parsed > 8192) return fallback;
    return (UINT)parsed;
}

static FLOAT parse_float_or_default(const wchar_t *value, FLOAT fallback) {
    if (value == nullptr || *value == L'\0') return fallback;
    wchar_t *end = nullptr;
    double parsed = wcstod(value, &end);
    if (end == value || parsed <= 0.0 || parsed > 8192.0) return fallback;
    return (FLOAT)parsed;
}

static void save_bgra_png(
    IWICImagingFactory *wic,
    const wchar_t *output_path,
    UINT width,
    UINT height,
    UINT stride,
    BYTE *pixels
) {
    HRESULT hr;
    IWICStream *stream = nullptr;
    IWICBitmapEncoder *encoder = nullptr;
    IWICBitmapFrameEncode *frame = nullptr;
    IPropertyBag2 *props = nullptr;
    WICPixelFormatGUID format = GUID_WICPixelFormat32bppPBGRA;

    hr = wic->CreateStream(&stream);
    check_hr(L"CreateStream", hr);
    hr = stream->InitializeFromFilename(output_path, GENERIC_WRITE);
    check_hr(L"InitializeFromFilename", hr);
    hr = wic->CreateEncoder(GUID_ContainerFormatPng, nullptr, &encoder);
    check_hr(L"CreateEncoder", hr);
    hr = encoder->Initialize((IStream *)stream, WICBitmapEncoderNoCache);
    check_hr(L"Encoder Initialize", hr);
    hr = encoder->CreateNewFrame(&frame, &props);
    check_hr(L"CreateNewFrame", hr);
    hr = frame->Initialize(props);
    check_hr(L"Frame Initialize", hr);
    hr = frame->SetSize(width, height);
    check_hr(L"Frame SetSize", hr);
    hr = frame->SetPixelFormat(&format);
    check_hr(L"Frame SetPixelFormat", hr);
    hr = frame->WritePixels(height, stride, stride * height, pixels);
    check_hr(L"Frame WritePixels", hr);
    hr = frame->Commit();
    check_hr(L"Frame Commit", hr);
    hr = encoder->Commit();
    check_hr(L"Encoder Commit", hr);

    SAFE_RELEASE(props);
    SAFE_RELEASE(frame);
    SAFE_RELEASE(encoder);
    SAFE_RELEASE(stream);
}

class ColorGlyphRenderer : public IDWriteTextRenderer {
public:
    ColorGlyphRenderer(ID2D1DeviceContext7 *context, IDWriteFactory8 *factory, ID2D1Brush *brush)
        : ref_count_(1), context_(context), factory_(factory), brush_(brush), logged_(false) {
        context_->AddRef();
        factory_->AddRef();
        brush_->AddRef();
    }

    virtual ~ColorGlyphRenderer() {
        SAFE_RELEASE(brush_);
        SAFE_RELEASE(factory_);
        SAFE_RELEASE(context_);
    }

    IFACEMETHOD(QueryInterface)(REFIID iid, void **object) override {
        if (object == nullptr) return E_POINTER;
        if (
            iid == __uuidof(IUnknown) ||
            iid == __uuidof(IDWritePixelSnapping) ||
            iid == __uuidof(IDWriteTextRenderer)
        ) {
            *object = static_cast<IDWriteTextRenderer *>(this);
            AddRef();
            return S_OK;
        }
        *object = nullptr;
        return E_NOINTERFACE;
    }

    IFACEMETHOD_(ULONG, AddRef)() override {
        return InterlockedIncrement(&ref_count_);
    }

    IFACEMETHOD_(ULONG, Release)() override {
        ULONG count = InterlockedDecrement(&ref_count_);
        if (count == 0) delete this;
        return count;
    }

    IFACEMETHOD(IsPixelSnappingDisabled)(void *, BOOL *isDisabled) override {
        *isDisabled = FALSE;
        return S_OK;
    }

    IFACEMETHOD(GetCurrentTransform)(void *, DWRITE_MATRIX *transform) override {
        transform->m11 = 1.0f;
        transform->m12 = 0.0f;
        transform->m21 = 0.0f;
        transform->m22 = 1.0f;
        transform->dx = 0.0f;
        transform->dy = 0.0f;
        return S_OK;
    }

    IFACEMETHOD(GetPixelsPerDip)(void *, FLOAT *pixelsPerDip) override {
        *pixelsPerDip = 1.0f;
        return S_OK;
    }

    IFACEMETHOD(DrawGlyphRun)(
        void *,
        FLOAT baselineOriginX,
        FLOAT baselineOriginY,
        DWRITE_MEASURING_MODE measuringMode,
        DWRITE_GLYPH_RUN const *glyphRun,
        DWRITE_GLYPH_RUN_DESCRIPTION const *glyphRunDescription,
        IUnknown *
    ) override {
        D2D1_POINT_2F origin = { baselineOriginX, baselineOriginY };
        DWRITE_GLYPH_IMAGE_FORMATS desiredFormats =
            (DWRITE_GLYPH_IMAGE_FORMATS)(
                DWRITE_GLYPH_IMAGE_FORMATS_TRUETYPE |
                DWRITE_GLYPH_IMAGE_FORMATS_CFF |
                DWRITE_GLYPH_IMAGE_FORMATS_COLR |
                DWRITE_GLYPH_IMAGE_FORMATS_SVG |
                DWRITE_GLYPH_IMAGE_FORMATS_PNG |
                DWRITE_GLYPH_IMAGE_FORMATS_JPEG |
                DWRITE_GLYPH_IMAGE_FORMATS_TIFF |
                DWRITE_GLYPH_IMAGE_FORMATS_PREMULTIPLIED_B8G8R8A8 |
                DWRITE_GLYPH_IMAGE_FORMATS_COLR_PAINT_TREE
            );
        DWRITE_MATRIX transform = { 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f };
        IDWriteColorGlyphRunEnumerator1 *enumerator = nullptr;
        HRESULT hr = factory_->TranslateColorGlyphRun(
            origin,
            glyphRun,
            glyphRunDescription,
            desiredFormats,
            context_->GetPaintFeatureLevel(),
            measuringMode,
            &transform,
            0,
            &enumerator
        );
        if (hr == DWRITE_E_NOCOLOR) {
            if (!logged_) {
                fwprintf(stderr, L"TranslateColorGlyphRun returned DWRITE_E_NOCOLOR; drawing monochrome fallback.\n");
                logged_ = true;
            }
            context_->DrawGlyphRun(origin, glyphRun, brush_, measuringMode);
            return S_OK;
        }
        if (FAILED(hr)) {
            return hr;
        }

        BOOL hasRun = FALSE;
        while (SUCCEEDED(enumerator->MoveNext(&hasRun)) && hasRun) {
            DWRITE_COLOR_GLYPH_RUN1 const *colorRun = nullptr;
            hr = enumerator->GetCurrentRun(&colorRun);
            if (FAILED(hr) || colorRun == nullptr) {
                SAFE_RELEASE(enumerator);
                return hr;
            }
            if (!logged_) {
                fwprintf(stderr, L"Color glyph format: 0x%08x paintLevel=%u\n",
                    (unsigned int)colorRun->glyphImageFormat,
                    (unsigned int)context_->GetPaintFeatureLevel());
            }

            D2D1_POINT_2F colorOrigin = { colorRun->baselineOriginX, colorRun->baselineOriginY };
            switch (colorRun->glyphImageFormat) {
                case DWRITE_GLYPH_IMAGE_FORMATS_COLR_PAINT_TREE:
                    context_->DrawPaintGlyphRun(
                        colorOrigin,
                        &colorRun->glyphRun,
                        brush_,
                        0,
                        colorRun->measuringMode
                    );
                    break;
                case DWRITE_GLYPH_IMAGE_FORMATS_SVG:
                    context_->DrawSvgGlyphRun(
                        colorOrigin,
                        &colorRun->glyphRun,
                        brush_,
                        nullptr,
                        0,
                        colorRun->measuringMode
                    );
                    break;
                case DWRITE_GLYPH_IMAGE_FORMATS_PNG:
                case DWRITE_GLYPH_IMAGE_FORMATS_JPEG:
                case DWRITE_GLYPH_IMAGE_FORMATS_TIFF:
                case DWRITE_GLYPH_IMAGE_FORMATS_PREMULTIPLIED_B8G8R8A8:
                    context_->DrawColorBitmapGlyphRun(
                        colorRun->glyphImageFormat,
                        colorOrigin,
                        &colorRun->glyphRun,
                        colorRun->measuringMode,
                        D2D1_COLOR_BITMAP_GLYPH_SNAP_OPTION_DEFAULT
                    );
                    break;
                default: {
                    ID2D1SolidColorBrush *layerBrush = nullptr;
                    ID2D1Brush *runBrush = brush_;
                    if (colorRun->paletteIndex != DWRITE_NO_PALETTE_INDEX) {
                        D2D1_COLOR_F color = {
                            colorRun->runColor.r,
                            colorRun->runColor.g,
                            colorRun->runColor.b,
                            colorRun->runColor.a
                        };
                        if (SUCCEEDED(context_->CreateSolidColorBrush(&color, nullptr, &layerBrush))) {
                            runBrush = layerBrush;
                        }
                    }
                    context_->DrawGlyphRun(colorOrigin, &colorRun->glyphRun, runBrush, colorRun->measuringMode);
                    SAFE_RELEASE(layerBrush);
                    break;
                }
            }
        }
        logged_ = true;
        SAFE_RELEASE(enumerator);
        return S_OK;
    }

    IFACEMETHOD(DrawUnderline)(
        void *,
        FLOAT,
        FLOAT,
        DWRITE_UNDERLINE const *,
        IUnknown *
    ) override {
        return S_OK;
    }

    IFACEMETHOD(DrawStrikethrough)(
        void *,
        FLOAT,
        FLOAT,
        DWRITE_STRIKETHROUGH const *,
        IUnknown *
    ) override {
        return S_OK;
    }

    IFACEMETHOD(DrawInlineObject)(
        void *,
        FLOAT,
        FLOAT,
        IDWriteInlineObject *,
        BOOL,
        BOOL,
        IUnknown *
    ) override {
        return S_OK;
    }

private:
    ULONG ref_count_;
    ID2D1DeviceContext7 *context_;
    IDWriteFactory8 *factory_;
    ID2D1Brush *brush_;
    bool logged_;
};

int wmain(int argc, wchar_t **argv) {
    if (argc < 3) {
        fwprintf(stderr, L"Usage: %ls <emoji-or-text> <output.png> [canvasSize=1024] [fontSize=720]\n", argv[0]);
        return 2;
    }

    const wchar_t *text = argv[1];
    const wchar_t *output_path = argv[2];
    UINT canvas_size = argc > 3 ? parse_uint_or_default(argv[3], 1024) : 1024;
    FLOAT font_size = argc > 4 ? parse_float_or_default(argv[4], 720.0f) : 720.0f;

    HRESULT hr = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
    check_hr(L"CoInitializeEx", hr);

    IWICImagingFactory *wic = nullptr;
    ID3D11Device *d3d_device = nullptr;
    ID3D11DeviceContext *d3d_context = nullptr;
    ID3D11Texture2D *render_texture = nullptr;
    ID3D11Texture2D *staging_texture = nullptr;
    IDXGIDevice *dxgi_device = nullptr;
    IDXGISurface *dxgi_surface = nullptr;
    ID2D1Factory7 *d2d_factory = nullptr;
    ID2D1Device6 *d2d_device = nullptr;
    ID2D1DeviceContext6 *base_d2d_context = nullptr;
    ID2D1DeviceContext7 *d2d_context = nullptr;
    ID2D1Bitmap1 *target_bitmap = nullptr;
    ID2D1SolidColorBrush *fallback_brush = nullptr;
    IDWriteFactory8 *dwrite_factory = nullptr;
    IDWriteTextFormat *format = nullptr;
    IDWriteTextLayout *layout = nullptr;
    ColorGlyphRenderer *renderer = nullptr;

    hr = CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&wic));
    check_hr(L"CoCreateInstance WIC", hr);

    D3D_FEATURE_LEVEL feature_levels[] = {
        D3D_FEATURE_LEVEL_11_1,
        D3D_FEATURE_LEVEL_11_0,
        D3D_FEATURE_LEVEL_10_1,
        D3D_FEATURE_LEVEL_10_0,
    };
    D3D_FEATURE_LEVEL feature_level;
    hr = D3D11CreateDevice(
        nullptr,
        D3D_DRIVER_TYPE_HARDWARE,
        nullptr,
        D3D11_CREATE_DEVICE_BGRA_SUPPORT,
        feature_levels,
        ARRAYSIZE(feature_levels),
        D3D11_SDK_VERSION,
        &d3d_device,
        &feature_level,
        &d3d_context
    );
    if (FAILED(hr)) {
        hr = D3D11CreateDevice(
            nullptr,
            D3D_DRIVER_TYPE_WARP,
            nullptr,
            D3D11_CREATE_DEVICE_BGRA_SUPPORT,
            feature_levels,
            ARRAYSIZE(feature_levels),
            D3D11_SDK_VERSION,
            &d3d_device,
            &feature_level,
            &d3d_context
        );
    }
    check_hr(L"D3D11CreateDevice", hr);

    D3D11_TEXTURE2D_DESC render_desc = {};
    render_desc.Width = canvas_size;
    render_desc.Height = canvas_size;
    render_desc.MipLevels = 1;
    render_desc.ArraySize = 1;
    render_desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    render_desc.SampleDesc.Count = 1;
    render_desc.Usage = D3D11_USAGE_DEFAULT;
    render_desc.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
    hr = d3d_device->CreateTexture2D(&render_desc, nullptr, &render_texture);
    check_hr(L"CreateTexture2D render", hr);

    D3D11_TEXTURE2D_DESC staging_desc = render_desc;
    staging_desc.Usage = D3D11_USAGE_STAGING;
    staging_desc.BindFlags = 0;
    staging_desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    hr = d3d_device->CreateTexture2D(&staging_desc, nullptr, &staging_texture);
    check_hr(L"CreateTexture2D staging", hr);

    hr = d3d_device->QueryInterface(IID_PPV_ARGS(&dxgi_device));
    check_hr(L"Query IDXGIDevice", hr);
    hr = render_texture->QueryInterface(IID_PPV_ARGS(&dxgi_surface));
    check_hr(L"Query IDXGISurface", hr);

    D2D1_FACTORY_OPTIONS factory_options = {};
    hr = D2D1CreateFactory(D2D1_FACTORY_TYPE_SINGLE_THREADED, factory_options, &d2d_factory);
    check_hr(L"D2D1CreateFactory ID2D1Factory7", hr);
    hr = d2d_factory->CreateDevice(dxgi_device, &d2d_device);
    check_hr(L"CreateDevice", hr);
    hr = d2d_device->CreateDeviceContext(D2D1_DEVICE_CONTEXT_OPTIONS_NONE, &base_d2d_context);
    check_hr(L"CreateDeviceContext6", hr);
    hr = base_d2d_context->QueryInterface(IID_PPV_ARGS(&d2d_context));
    check_hr(L"Query ID2D1DeviceContext7", hr);

    D2D1_BITMAP_PROPERTIES1 bitmap_props = {};
    bitmap_props.pixelFormat.format = DXGI_FORMAT_B8G8R8A8_UNORM;
    bitmap_props.pixelFormat.alphaMode = D2D1_ALPHA_MODE_PREMULTIPLIED;
    bitmap_props.dpiX = 96.0f;
    bitmap_props.dpiY = 96.0f;
    bitmap_props.bitmapOptions = D2D1_BITMAP_OPTIONS_TARGET;
    hr = d2d_context->CreateBitmapFromDxgiSurface(dxgi_surface, &bitmap_props, &target_bitmap);
    check_hr(L"CreateBitmapFromDxgiSurface", hr);
    d2d_context->SetTarget(target_bitmap);

    hr = DWriteCreateFactory(DWRITE_FACTORY_TYPE_SHARED, __uuidof(IDWriteFactory8), (IUnknown **)&dwrite_factory);
    check_hr(L"DWriteCreateFactory8", hr);
    hr = dwrite_factory->CreateTextFormat(
        L"Segoe UI Emoji",
        nullptr,
        DWRITE_FONT_WEIGHT_NORMAL,
        DWRITE_FONT_STYLE_NORMAL,
        DWRITE_FONT_STRETCH_NORMAL,
        font_size,
        L"en-us",
        &format
    );
    check_hr(L"CreateTextFormat", hr);
    check_hr(L"SetTextAlignment", format->SetTextAlignment(DWRITE_TEXT_ALIGNMENT_CENTER));
    check_hr(L"SetParagraphAlignment", format->SetParagraphAlignment(DWRITE_PARAGRAPH_ALIGNMENT_CENTER));
    hr = dwrite_factory->CreateTextLayout(
        text,
        (UINT32)wcslen(text),
        format,
        (FLOAT)canvas_size,
        (FLOAT)canvas_size,
        &layout
    );
    check_hr(L"CreateTextLayout", hr);

    D2D1_COLOR_F clear_color = { 0.0f, 0.0f, 0.0f, 0.0f };
    D2D1_COLOR_F brush_color = { 0.0f, 0.0f, 0.0f, 1.0f };
    D2D1_RECT_F rect = { 0.0f, 0.0f, (FLOAT)canvas_size, (FLOAT)canvas_size };
    hr = d2d_context->CreateSolidColorBrush(&brush_color, nullptr, &fallback_brush);
    check_hr(L"CreateSolidColorBrush", hr);
    renderer = new ColorGlyphRenderer(d2d_context, dwrite_factory, fallback_brush);

    d2d_context->BeginDraw();
    d2d_context->Clear(clear_color);
    hr = layout->Draw(nullptr, renderer, 0.0f, 0.0f);
    check_hr(L"Layout Draw", hr);
    hr = d2d_context->EndDraw();
    check_hr(L"EndDraw", hr);

    d3d_context->CopyResource(staging_texture, render_texture);
    D3D11_MAPPED_SUBRESOURCE mapped = {};
    hr = d3d_context->Map(staging_texture, 0, D3D11_MAP_READ, 0, &mapped);
    check_hr(L"Map staging", hr);

    BYTE *packed = (BYTE *)malloc(canvas_size * canvas_size * 4);
    if (!packed) {
        fwprintf(stderr, L"Out of memory.\n");
        return 1;
    }
    for (UINT y = 0; y < canvas_size; ++y) {
        memcpy(
            packed + y * canvas_size * 4,
            (BYTE *)mapped.pData + y * mapped.RowPitch,
            canvas_size * 4
        );
    }
    d3d_context->Unmap(staging_texture, 0);

    save_bgra_png(wic, output_path, canvas_size, canvas_size, canvas_size * 4, packed);
    free(packed);

    SAFE_RELEASE(renderer);
    SAFE_RELEASE(layout);
    SAFE_RELEASE(format);
    SAFE_RELEASE(fallback_brush);
    SAFE_RELEASE(target_bitmap);
    SAFE_RELEASE(d2d_context);
    SAFE_RELEASE(base_d2d_context);
    SAFE_RELEASE(d2d_device);
    SAFE_RELEASE(d2d_factory);
    SAFE_RELEASE(dxgi_surface);
    SAFE_RELEASE(dxgi_device);
    SAFE_RELEASE(staging_texture);
    SAFE_RELEASE(render_texture);
    SAFE_RELEASE(d3d_context);
    SAFE_RELEASE(d3d_device);
    SAFE_RELEASE(dwrite_factory);
    SAFE_RELEASE(wic);
    CoUninitialize();
    return 0;
}
