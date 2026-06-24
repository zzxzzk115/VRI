# Flight Helmet

## License Information

Donated by Microsoft for glTF testing

[![CC0](http://i.creativecommons.org/p/zero/1.0/88x31.png)](http://creativecommons.org/publicdomain/zero/1.0/)  
To the extent possible under law, Microsoft has waived all copyright and related or neighboring rights to this asset.

## Modifications in this repository

This copy is trimmed for the VRI examples (examples/raytracing, examples/pathtracer):

- Textures downscaled to 1024x1024 (originals were up to 2048) — ample for the examples' resolution.
- The 5 normal maps were removed (the example shaders sample only baseColor + occlusion); the
  glTF's `normalTexture` references and the corresponding `images`/`textures` entries were dropped
  and the remaining indices remapped. Total size ~14 MB (was ~52 MB).

The model is CC0 (see the attribution above); these edits are purely size reductions.
