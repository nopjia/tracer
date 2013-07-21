# CUDA Path Tracer

Advanced Rendering Seminar, University of Pennsylvania, Spring 2013

## Details

Real-time iterative path tracer implemented on CUDA.

Path tracing kernel is a per-bounce structure, instead of a mega-kernel structure, in order to minimize thread divergence.

Image draw done through CUDA OpenGL Interop, by mapping OpenGL pixel buffer object to CUDA memory, then drawing the pixel buffer out as a texture on a full screen quad.

Runs at ~20fps on GeForce GT 650M. Achieves considerably converged image at ~1000 iterations, after ~50 seconds.

## Features

- Diffuse Shading
- Diffuse Reflection
- Specular Reflection
- Fresnel Refraction
- Anti-aliasing
- *Interactive* Depth of Field
- *Interactive* Camera
- *Interactive* Object Manipulation


- Two visualization modes:
    - Mode 1: fast single bounce ray tracing (default at startup)
    - Mode 2: path tracing

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

## References

- [Matt Pharr, Physically Based Rendering](http://www.pbrt.org/)
- [CUDA Ray Tracing Tutorial](http://cg.alexandra.dk/2009/08/10/triers-cuda-ray-tracing-tutorial/)
- [smallpt: Global Illumination in 99 lines of C++](http://www.kevinbeason.com/smallpt/)
- [A Raytracer in C++](http://www.codermind.com/articles/Raytracer-in-C++-Introduction-What-is-ray-tracing.html)
- [Reflections and Refractions in Ray Tracing](http://graphics.stanford.edu/courses/cs148-10-summer/docs/2006--degreve--reflection_refraction.pdf)