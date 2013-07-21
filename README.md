# CUDA Path Tracer

Interactive real-time iterative path tracer in CUDA.

Advanced Rendering Seminar, University of Pennsylvania, Spring 2013

![render](https://photos-1.dropbox.com/t/0/AACdWxeqFN2jJNrI21odXSl9qw5B3WFONW1oIP_eEKH6sQ/12/5058905/jpeg/2048x1536/3/1374397200/0/2/tracer%202013-07-21%2014-53-21-02.bmp/mI2kCnCJANTnk8by_DXcOj2aTXSe1hmOYvFezyry5mQ)

## Features

- Diffuse Shading
- Diffuse Reflection
- Specular Reflection
- Fresnel Reflection and Refraction
- Anti-aliasing
- *Interactive* Depth of Field
- *Interactive* Camera
- *Interactive* Object Manipulation

Two visualization modes:

1. Fast single bounce ray tracing (default at startup)
1. Path tracing

## Controls

Modes

- **Ray Tracer** - Press 1
- **Path Tracer** - Press 2

Camera Controls

- **Rotate** - Left Click drag
- **Zoom** - Right Click drag
- **Pan** - Middle Click drag
- **Focal Length** - F / Shift+F
- **Lens Radius** - G / Shift+G

Object Manipulation

- **Translate** - Ctrl + Left Click +  drag
- **Scale** - Ctrl + Right Click +  drag
- **Rotate** - Ctrl + Middle Click +  drag

## Details

Interactive real-time iterative path tracer in CUDA.

Image draw done through CUDA OpenGL Interop, by mapping OpenGL pixel buffer object to CUDA memory, then drawing the pixel buffer out as a texture on a full screen quad.

Path tracing kernel is a per-bounce structure, instead of a mega-kernel structure, in order to minimize thread divergence. 

Stream compaction was attempted in order discard dead paths and minimize thread workload. However, the overhead costs and memory coherency issues turned out to negatively affect performance. (If interested, see corresponding [commit](https://github.com/nopjia/tracer/commit/9df5f0df5b0878ef1253b34402f037b2977c55ed).)

In addition, there is a fast visualization mode for quickly viewing the scene before path tracing. Instead of OpenGL draw, it uses single bounce ray tracing on CUDA, in order to ensure exact image pixel correspondence with path tracing.

## Performance

Hardware:
Intel Core i7 2.40GHz 8.00GB RAM
NVIDIA GeForce GT 650M

At 256x256 buffer size:
~20fps. Approximately converges at ~400 iterations, ~20 seconds.

At 512x512 buffer size:
~10fps. Approximately converges at ~300 iterations, ~30 seconds.

## Media

[Images](https://www.dropbox.com/sh/s84z7zrgsmnzt5p/IUVtvwegdP#/)

## References

- [Matt Pharr, Physically Based Rendering](http://www.pbrt.org/)
- [CUDA Ray Tracing Tutorial](http://cg.alexandra.dk/2009/08/10/triers-cuda-ray-tracing-tutorial/)
- [smallpt: Global Illumination in 99 lines of C++](http://www.kevinbeason.com/smallpt/)
- [A Raytracer in C++](http://www.codermind.com/articles/Raytracer-in-C++-Introduction-What-is-ray-tracing.html)
- [Reflections and Refractions in Ray Tracing](http://graphics.stanford.edu/courses/cs148-10-summer/docs/2006--degreve--reflection_refraction.pdf)