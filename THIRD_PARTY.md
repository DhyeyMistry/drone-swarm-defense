# Third-party assets

## Tello URDF

`assets/tello/tello.urdf` is derived from `tello_description/urdf/tello.xml` in
[tello_ros](https://github.com/clydemcqueen/tello_ros) by Clyde McQueen and Peter Mullen,
licensed BSD-3-Clause. The unmodified upstream file is kept alongside it as
`tello_orig_upstream.xml`.

Changes made for this kit:

- removed the Gazebo-specific `<gazebo>` plugin blocks (`TelloPlugin`, `libgazebo_ros_camera.so`)
- resolved the `${suffix}` and `${topic_ns}` placeholders
- corrected the physical properties to Tello EDU spec: mass 0.087 kg (was 0.1 kg),
  collision box 98 x 92.5 x 41 mm (was 180 x 180 x 50 mm), inertia recomputed to match

### BSD 3-Clause License

Copyright (c) Clyde McQueen and Peter Mullen.

Redistribution and use in source and binary forms, with or without modification, are
permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of
   conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of
   conditions and the following disclaimer in the documentation and/or other materials
   provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors may be used
   to endorse or promote products derived from this software without specific prior
   written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
