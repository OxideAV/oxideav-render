//! Procedural reference scenes and image metrics shared by every
//! backend's tests (CPU scanline / raycast / path tracer, the wgpu
//! backend), so all of them can be checked against the same inputs.
//!
//! Every scene is self-contained: geometry, materials, textures (in
//! the built-in [`crate::texture::RAW_RGBA8_MIME`] container, so no
//! image codec is needed), `KHR_lights_punctual` lights, and one scene
//! camera — render with `RenderOptions { scene_camera: Some(0), .. }`
//! to frame them as designed.
//!
//! The metrics ([`mean_abs_error`], [`psnr`], [`region_mean`],
//! [`region_stddev`]) support tolerance-based comparisons — backends
//! differ in sampling / filtering details, so tests assert properties
//! ("the left wall is red", "a smooth sphere's highlight is brighter
//! than a rough one's") or bounded error against small goldens rather
//! than byte equality.

use std::f32::consts::{FRAC_PI_2, PI};
use std::sync::Arc;

use oxideav_mesh3d::{
    AlphaMode, Animation, AnimationProperty, AnimationSampler, AnimationValues, Camera,
    InMemoryAsset, Indices, Interpolation, Light, Material, Mesh, MorphTarget, Node, NodeId,
    Primitive, Sampler, Scene3D, Skeleton, Skin, Texture, TextureRef, Topology, Transform,
};

use crate::image::RgbaImage;
use crate::texture::{encode_raw_rgba8, RAW_RGBA8_MIME};

// ---------------------------------------------------------------------
// Geometry builders.
// ---------------------------------------------------------------------

/// Planar quad `p0 → p1 → p2 → p3` (counter-clockwise seen from the
/// front), with flat normals, tangents along `p0 → p1`, and UVs
/// `p0 = (0, 1)`, `p1 = (1, 1)`, `p2 = (1, 0)`, `p3 = (0, 0)` scaled
/// by `uv_scale`.
pub fn quad(p: [[f32; 3]; 4], uv_scale: [f32; 2]) -> Primitive {
    let sub = |a: [f32; 3], b: [f32; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    let e1 = sub(p[1], p[0]);
    let e2 = sub(p[3], p[0]);
    let n = normalize([
        e1[1] * e2[2] - e1[2] * e2[1],
        e1[2] * e2[0] - e1[0] * e2[2],
        e1[0] * e2[1] - e1[1] * e2[0],
    ]);
    let t = normalize(e1);
    let mut prim = Primitive::new(Topology::Triangles);
    prim.positions = p.to_vec();
    prim.normals = Some(vec![n; 4]);
    prim.tangents = Some(vec![[t[0], t[1], t[2], 1.0]; 4]);
    let (su, sv) = (uv_scale[0], uv_scale[1]);
    prim.uvs = vec![vec![[0.0, sv], [su, sv], [su, 0.0], [0.0, 0.0]]];
    prim.indices = Some(Indices::U32(vec![0, 1, 2, 0, 2, 3]));
    prim
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if l > 0.0 {
        [v[0] / l, v[1] / l, v[2] / l]
    } else {
        v
    }
}

/// Axis-aligned box `min..max` with outward faces (one primitive,
/// 24 vertices, flat normals, per-face UVs).
pub fn cuboid(min: [f32; 3], max: [f32; 3]) -> Primitive {
    let [x0, y0, z0] = min;
    let [x1, y1, z1] = max;
    let faces = [
        [[x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], // +Z
        [[x1, y0, z0], [x0, y0, z0], [x0, y1, z0], [x1, y1, z0]], // -Z
        [[x1, y0, z1], [x1, y0, z0], [x1, y1, z0], [x1, y1, z1]], // +X
        [[x0, y0, z0], [x0, y0, z1], [x0, y1, z1], [x0, y1, z0]], // -X
        [[x0, y1, z1], [x1, y1, z1], [x1, y1, z0], [x0, y1, z0]], // +Y
        [[x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1]], // -Y
    ];
    merge(faces.iter().map(|f| quad(*f, [1.0, 1.0])).collect())
}

/// Concatenate triangle primitives (attributes must match in kind).
fn merge(prims: Vec<Primitive>) -> Primitive {
    let mut out = Primitive::new(Topology::Triangles);
    let mut normals = Vec::new();
    let mut tangents = Vec::new();
    let mut uvs = Vec::new();
    let mut idx = Vec::new();
    for p in prims {
        let base = out.positions.len() as u32;
        if let Some(Indices::U32(i)) = &p.indices {
            idx.extend(i.iter().map(|v| v + base));
        }
        out.positions.extend_from_slice(&p.positions);
        normals.extend_from_slice(p.normals.as_deref().unwrap_or(&[]));
        tangents.extend_from_slice(p.tangents.as_deref().unwrap_or(&[]));
        uvs.extend_from_slice(p.uvs.first().map(Vec::as_slice).unwrap_or(&[]));
    }
    out.normals = Some(normals);
    out.tangents = Some(tangents);
    out.uvs = vec![uvs];
    out.indices = Some(Indices::U32(idx));
    out
}

/// UV sphere of `radius` centred at the origin: smooth normals,
/// tangents along increasing `u`, UVs (`u` around, `v` pole to pole).
pub fn uv_sphere(radius: f32, segments: u32, rings: u32) -> Primitive {
    let segments = segments.max(3);
    let rings = rings.max(2);
    let mut prim = Primitive::new(Topology::Triangles);
    let mut normals = Vec::new();
    let mut tangents = Vec::new();
    let mut uvs = Vec::new();
    for r in 0..=rings {
        let v = r as f32 / rings as f32;
        let theta = v * PI;
        for s in 0..=segments {
            let u = s as f32 / segments as f32;
            let phi = u * 2.0 * PI;
            let n = [
                theta.sin() * phi.cos(),
                theta.cos(),
                -theta.sin() * phi.sin(),
            ];
            prim.positions
                .push([n[0] * radius, n[1] * radius, n[2] * radius]);
            normals.push(n);
            tangents.push([-phi.sin(), 0.0, -phi.cos(), 1.0]);
            uvs.push([u, v]);
        }
    }
    let row = segments + 1;
    let mut idx = Vec::new();
    for r in 0..rings {
        for s in 0..segments {
            let a = r * row + s;
            let b = a + row;
            idx.extend_from_slice(&[a, b, a + 1, a + 1, b, b + 1]);
        }
    }
    prim.normals = Some(normals);
    prim.tangents = Some(tangents);
    prim.uvs = vec![uvs];
    prim.indices = Some(Indices::U32(idx));
    prim
}

/// `size × size` RGBA8 checkerboard of `cells × cells` squares
/// alternating `a` / `b` (top-left cell = `a`).
pub fn checker_rgba8(size: u32, cells: u32, a: [u8; 4], b: [u8; 4]) -> Vec<u8> {
    let cell = (size / cells.max(1)).max(1);
    let mut out = Vec::with_capacity((size * size * 4) as usize);
    for y in 0..size {
        for x in 0..size {
            let c = if ((x / cell) + (y / cell)) % 2 == 0 {
                a
            } else {
                b
            };
            out.extend_from_slice(&c);
        }
    }
    out
}

/// A texture backed by the built-in raw RGBA8 container.
pub fn raw_texture(width: u32, height: u32, rgba: &[u8], sampler: Sampler) -> Texture {
    let mut t = Texture::from_source(Arc::new(InMemoryAsset::new(
        Some(RAW_RGBA8_MIME.to_string()),
        encode_raw_rgba8(width, height, rgba),
    )));
    t.sampler = sampler;
    t
}

/// Dielectric material of linear `rgb`, given roughness.
pub fn diffuse_material(rgb: [f32; 3], roughness: f32) -> Material {
    let mut m = Material::new();
    m.base_color = [rgb[0], rgb[1], rgb[2], 1.0];
    m.metallic = 0.0;
    m.roughness = roughness;
    m
}

fn add_mesh_node(scene: &mut Scene3D, prim: Primitive, transform: Transform) -> NodeId {
    let m = scene.add_mesh(Mesh::new(None).with_primitive(prim));
    let n = scene.add_node(Node::new().with_mesh(m).with_transform(transform));
    scene.add_root(n);
    n
}

fn translate(t: [f32; 3]) -> Transform {
    Transform::Trs {
        translation: t,
        rotation: [0.0, 0.0, 0.0, 1.0],
        scale: [1.0; 3],
    }
}

/// Quaternion (xyzw) rotating `angle` radians about unit `axis`.
pub fn quat_axis_angle(axis: [f32; 3], angle: f32) -> [f32; 4] {
    let (s, c) = (angle * 0.5).sin_cos();
    let a = normalize(axis);
    [a[0] * s, a[1] * s, a[2] * s, c]
}

/// Quaternion orienting a node so its local `-Z` points along `dir`
/// (glTF cameras / lights look down `-Z`), keeping `+Y` up-ish.
pub fn look_rotation(dir: [f32; 3]) -> [f32; 4] {
    let f = normalize(dir);
    let up = if f[1].abs() > 0.99 {
        [0.0, 0.0, -1.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    // Basis: x = up × z, y = z × x, z = -f.
    let z = [-f[0], -f[1], -f[2]];
    let x = normalize([
        up[1] * z[2] - up[2] * z[1],
        up[2] * z[0] - up[0] * z[2],
        up[0] * z[1] - up[1] * z[0],
    ]);
    let y = [
        z[1] * x[2] - z[2] * x[1],
        z[2] * x[0] - z[0] * x[2],
        z[0] * x[1] - z[1] * x[0],
    ];
    // Rotation matrix columns x, y, z → quaternion.
    let (m00, m11, m22) = (x[0], y[1], z[2]);
    let tr = m00 + m11 + m22;
    if tr > 0.0 {
        let s = (tr + 1.0).sqrt() * 2.0;
        [
            (y[2] - z[1]) / s,
            (z[0] - x[2]) / s,
            (x[1] - y[0]) / s,
            0.25 * s,
        ]
    } else if m00 > m11 && m00 > m22 {
        let s = (1.0 + m00 - m11 - m22).sqrt() * 2.0;
        [
            0.25 * s,
            (y[0] + x[1]) / s,
            (z[0] + x[2]) / s,
            (y[2] - z[1]) / s,
        ]
    } else if m11 > m22 {
        let s = (1.0 + m11 - m00 - m22).sqrt() * 2.0;
        [
            (y[0] + x[1]) / s,
            0.25 * s,
            (z[1] + y[2]) / s,
            (z[0] - x[2]) / s,
        ]
    } else {
        let s = (1.0 + m22 - m00 - m11).sqrt() * 2.0;
        [
            (z[0] + x[2]) / s,
            (z[1] + y[2]) / s,
            0.25 * s,
            (x[1] - y[0]) / s,
        ]
    }
}

/// Add a perspective camera node at `eye` looking at `target`.
pub fn add_camera(scene: &mut Scene3D, eye: [f32; 3], target: [f32; 3], yfov: f32) -> NodeId {
    let c = scene.add_camera(Camera::perspective(yfov, 0.05));
    let dir = [target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]];
    let mut node = Node::new().with_transform(Transform::Trs {
        translation: eye,
        rotation: look_rotation(dir),
        scale: [1.0; 3],
    });
    node.camera = Some(c);
    let n = scene.add_node(node);
    scene.add_root(n);
    n
}

/// Add a light node at `position` pointing along `dir`.
pub fn add_light(scene: &mut Scene3D, light: Light, position: [f32; 3], dir: [f32; 3]) -> NodeId {
    let l = scene.add_light(light);
    let mut node = Node::new().with_transform(Transform::Trs {
        translation: position,
        rotation: look_rotation(dir),
        scale: [1.0; 3],
    });
    node.light = Some(l);
    let n = scene.add_node(node);
    scene.add_root(n);
    n
}

// ---------------------------------------------------------------------
// Reference scenes.
// ---------------------------------------------------------------------

/// Cornell-box-like room: `[-1, 1]³`, open toward `+Z`; red left wall,
/// green right wall, white floor / ceiling / back wall, a tall and a
/// short white block, a point light under the ceiling, camera at
/// `(0, 0, 3.4)` looking down `-Z`.
pub fn cornell_box() -> Scene3D {
    let mut scene = Scene3D::new();
    let white = scene.add_material(diffuse_material([0.75, 0.75, 0.75], 1.0));
    let red = scene.add_material(diffuse_material([0.65, 0.05, 0.05], 1.0));
    let green = scene.add_material(diffuse_material([0.12, 0.45, 0.15], 1.0));
    let walls: [([[f32; 3]; 4], _); 5] = [
        // Floor (faces +Y).
        (
            [
                [-1.0, -1.0, 1.0],
                [1.0, -1.0, 1.0],
                [1.0, -1.0, -1.0],
                [-1.0, -1.0, -1.0],
            ],
            white,
        ),
        // Ceiling (faces -Y).
        (
            [
                [-1.0, 1.0, -1.0],
                [1.0, 1.0, -1.0],
                [1.0, 1.0, 1.0],
                [-1.0, 1.0, 1.0],
            ],
            white,
        ),
        // Back wall (faces +Z).
        (
            [
                [-1.0, -1.0, -1.0],
                [1.0, -1.0, -1.0],
                [1.0, 1.0, -1.0],
                [-1.0, 1.0, -1.0],
            ],
            white,
        ),
        // Left wall (x = -1, faces +X).
        (
            [
                [-1.0, -1.0, 1.0],
                [-1.0, -1.0, -1.0],
                [-1.0, 1.0, -1.0],
                [-1.0, 1.0, 1.0],
            ],
            red,
        ),
        // Right wall (x = 1, faces -X).
        (
            [
                [1.0, -1.0, -1.0],
                [1.0, -1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, -1.0],
            ],
            green,
        ),
    ];
    for (q, mat) in walls {
        let mut p = quad(q, [1.0, 1.0]);
        p.material = Some(mat);
        add_mesh_node(&mut scene, p, Transform::identity());
    }
    let mut tall = cuboid([-0.6, -1.0, -0.6], [-0.1, 0.2, -0.1]);
    tall.material = Some(white);
    add_mesh_node(&mut scene, tall, Transform::identity());
    let mut short = cuboid([0.15, -1.0, -0.1], [0.65, -0.4, 0.4]);
    short.material = Some(white);
    add_mesh_node(&mut scene, short, Transform::identity());
    add_light(
        &mut scene,
        Light::Point {
            color: [1.0, 0.95, 0.85],
            intensity: 6.0,
            range: None,
        },
        [0.0, 0.85, 0.0],
        [0.0, -1.0, 0.0],
    );
    add_camera(&mut scene, [0.0, 0.0, 3.4], [0.0, 0.0, 0.0], 0.75);
    scene
}

/// `cols × rows` grid of spheres: metallic rises `0 → 1` left to
/// right, roughness rises `0.05 → 1` bottom to top. Sphere `(c, r)` is
/// node `c + r·cols` and sits at `x = (c − (cols−1)/2)·2.2`,
/// `y = (r − (rows−1)/2)·2.2`. One directional light from the upper
/// front-left; camera on `+Z`.
pub fn sphere_grid(cols: u32, rows: u32) -> Scene3D {
    let cols = cols.max(1);
    let rows = rows.max(1);
    let mut scene = Scene3D::new();
    let sphere = uv_sphere(1.0, 48, 24);
    for r in 0..rows {
        for c in 0..cols {
            let mut m = Material::new();
            m.base_color = [0.9, 0.55, 0.25, 1.0];
            m.metallic = if cols > 1 {
                c as f32 / (cols - 1) as f32
            } else {
                0.0
            };
            m.roughness = if rows > 1 {
                0.05 + 0.95 * r as f32 / (rows - 1) as f32
            } else {
                0.5
            };
            let mat = scene.add_material(m);
            let mut p = sphere.clone();
            p.material = Some(mat);
            let x = (c as f32 - (cols - 1) as f32 * 0.5) * 2.2;
            let y = (r as f32 - (rows - 1) as f32 * 0.5) * 2.2;
            add_mesh_node(&mut scene, p, translate([x, y, 0.0]));
        }
    }
    add_light(
        &mut scene,
        Light::Directional {
            color: [1.0; 3],
            intensity: 3.0,
        },
        [0.0; 3],
        [0.4, -0.5, -1.0],
    );
    let extent = (cols.max(rows) as f32) * 2.2;
    add_camera(&mut scene, [0.0, 0.0, extent * 1.6], [0.0; 3], 0.8);
    scene
}

/// A long checkerboard floor (`x ∈ [-2, 2]`, `z ∈ [-30, 2]`, `y = 0`,
/// 64×64 texture of 8×8 cells, UVs repeating 2× across and 16× along)
/// seen from a low camera — exercises perspective-correct UVs and
/// minification. `sampler` controls filtering / wrap. Unlit, so texel
/// values reach the framebuffer untouched.
pub fn checker_floor(sampler: Sampler) -> Scene3D {
    let mut scene = Scene3D::new();
    let tex = scene.add_texture(raw_texture(
        64,
        64,
        &checker_rgba8(64, 8, [255, 255, 255, 255], [0, 0, 0, 255]),
        sampler,
    ));
    let mut m = Material::new();
    m.base_color_texture = Some(TextureRef::new(tex));
    m.ext.unlit = true;
    m.double_sided = true;
    let mat = scene.add_material(m);
    let mut p = quad(
        [
            [-2.0, 0.0, 2.0],
            [2.0, 0.0, 2.0],
            [2.0, 0.0, -30.0],
            [-2.0, 0.0, -30.0],
        ],
        [2.0, 16.0],
    );
    p.material = Some(mat);
    add_mesh_node(&mut scene, p, Transform::identity());
    add_camera(&mut scene, [0.0, 0.6, 2.5], [0.0, 0.0, -6.0], 0.9);
    scene
}

/// A camera-facing unit quad (`[-1, 1]²` at `z = 0`) carrying an unlit
/// 4×4 checker texture (cells: red / blue), camera at `z = 2.5`.
pub fn textured_quad(sampler: Sampler) -> Scene3D {
    let mut scene = Scene3D::new();
    let tex = scene.add_texture(raw_texture(
        4,
        4,
        &checker_rgba8(4, 4, [255, 0, 0, 255], [0, 0, 255, 255]),
        sampler,
    ));
    let mut m = Material::new();
    m.base_color_texture = Some(TextureRef::new(tex));
    m.ext.unlit = true;
    let mat = scene.add_material(m);
    let mut p = quad(
        [
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0],
        ],
        [1.0, 1.0],
    );
    p.material = Some(mat);
    add_mesh_node(&mut scene, p, Transform::identity());
    add_camera(&mut scene, [0.0, 0.0, 2.5], [0.0; 3], 0.9);
    scene
}

/// Alpha-mode test: an unlit opaque blue back plane (`z = -1`), an
/// unlit green MASK plane (`z = 0`, cutoff 0.5, alpha from a 2×2-cell
/// checker: top-left and bottom-right cells opaque, the others
/// transparent), and an unlit red BLEND plane (`z = 0.5`, alpha 0.5)
/// covering only the left half (`x < 0`). All planes span
/// `[-1, 1]²`; camera on `+Z` with a 90° vertical FOV at distance 1.5
/// (so the planes fill the view).
pub fn alpha_planes() -> Scene3D {
    let mut scene = Scene3D::new();
    let unlit = |rgb: [f32; 3], a: f32, mode: AlphaMode| {
        let mut m = Material::new();
        m.base_color = [rgb[0], rgb[1], rgb[2], a];
        m.ext.unlit = true;
        m.alpha_mode = mode;
        m
    };
    let blue = scene.add_material(unlit([0.0, 0.0, 1.0], 1.0, AlphaMode::Opaque));
    let mask_tex = scene.add_texture(raw_texture(
        2,
        2,
        &[
            255, 255, 255, 255, 255, 255, 255, 0, //
            255, 255, 255, 0, 255, 255, 255, 255,
        ],
        Sampler::default_sampler()
            .with_mag_filter(oxideav_mesh3d::MagFilter::Nearest)
            .with_min_filter(oxideav_mesh3d::MinFilter::Nearest),
    ));
    let mut gm = unlit([0.0, 1.0, 0.0], 1.0, AlphaMode::Mask { cutoff: 0.5 });
    gm.base_color_texture = Some(TextureRef::new(mask_tex));
    let green = scene.add_material(gm);
    let red = scene.add_material(unlit([1.0, 0.0, 0.0], 0.5, AlphaMode::Blend));
    let plane = |z: f32, x1: f32| {
        quad(
            [[-1.0, -1.0, z], [x1, -1.0, z], [x1, 1.0, z], [-1.0, 1.0, z]],
            [1.0, 1.0],
        )
    };
    for (z, x1, mat) in [(-1.0, 1.0, blue), (0.0, 1.0, green), (0.5, 0.0, red)] {
        let mut p = plane(z, x1);
        p.material = Some(mat);
        add_mesh_node(&mut scene, p, Transform::identity());
    }
    add_camera(&mut scene, [0.0, 0.0, 1.5], [0.0; 3], FRAC_PI_2);
    scene
}

/// Shadow test: a white floor (`y = 0`, `[-3, 3]²`), a white cube
/// (`[-0.5, 0.5]` × `[1, 2]` × `[-0.5, 0.5]`) floating above it, a
/// straight-down directional light and a camera looking down at the
/// floor from the front. The floor point `(0, 0, 0)` is in the cube's
/// shadow; `(2, 0, 2)` is lit.
pub fn shadow_box() -> Scene3D {
    let mut scene = Scene3D::new();
    let white = scene.add_material(diffuse_material([0.8, 0.8, 0.8], 0.9));
    let mut floor = quad(
        [
            [-3.0, 0.0, 3.0],
            [3.0, 0.0, 3.0],
            [3.0, 0.0, -3.0],
            [-3.0, 0.0, -3.0],
        ],
        [1.0, 1.0],
    );
    floor.material = Some(white);
    add_mesh_node(&mut scene, floor, Transform::identity());
    let mut cube = cuboid([-0.5, 1.0, -0.5], [0.5, 2.0, 0.5]);
    cube.material = Some(white);
    add_mesh_node(&mut scene, cube, Transform::identity());
    add_light(
        &mut scene,
        Light::Directional {
            color: [1.0; 3],
            intensity: 3.0,
        },
        [0.0; 3],
        [0.0, -1.0, 0.0],
    );
    add_camera(&mut scene, [0.0, 6.0, 6.0], [0.0, 0.0, 0.0], 0.9);
    scene
}

/// Skinned + morphed animated beam: a `2 × 0.2` strip along `+X`
/// (21 vertex columns) bound to two joints (root at the origin,
/// `elbow` at `x = 1`, linear weights over `x ∈ [0.5, 1.5]`), with one
/// morph target lifting every vertex by `+0.3 Y`. Animation 0 rotates
/// the elbow `0 → 90°` about `+Z` and drives the morph weight
/// `0 → 1` over `t ∈ [0, 1]`. Camera on `+Z`.
///
/// At `t = 0` the tip sits at `(2, 0)`; at `t = 1` it sits at roughly
/// `(1, 1.3)`.
pub fn skinned_morph_beam() -> Scene3D {
    let mut scene = Scene3D::new();
    let cols = 21u32;
    let mut prim = Primitive::new(Topology::Triangles);
    let mut joints = Vec::new();
    let mut weights = Vec::new();
    for c in 0..cols {
        let x = 2.0 * c as f32 / (cols - 1) as f32;
        let w1 = ((x - 0.5) / 1.0).clamp(0.0, 1.0);
        for y in [-0.1f32, 0.1] {
            prim.positions.push([x, y, 0.0]);
            joints.push([0u16, 1, 0, 0]);
            weights.push([1.0 - w1, w1, 0.0, 0.0]);
        }
    }
    let mut idx = Vec::new();
    for c in 0..cols - 1 {
        let a = 2 * c;
        idx.extend_from_slice(&[a, a + 2, a + 3, a, a + 3, a + 1]);
    }
    prim.indices = Some(Indices::U32(idx));
    prim.normals = Some(vec![[0.0, 0.0, 1.0]; prim.positions.len()]);
    prim.joints = Some(joints);
    prim.weights = Some(weights);
    prim.targets = vec![MorphTarget::with_deltas(
        Some(vec![[0.0, 0.3, 0.0]; prim.positions.len()]),
        None,
        None,
    )];
    let mat = scene.add_material(diffuse_material([0.2, 0.6, 0.9], 0.6));
    prim.material = Some(mat);
    let mesh = scene.add_mesh(Mesh::new(None).with_primitive(prim).with_weights(vec![0.0]));

    // Joint hierarchy: root (origin) → elbow (x = 1).
    let elbow = scene.add_node(
        Node::new()
            .with_name("elbow")
            .with_transform(translate([1.0, 0.0, 0.0])),
    );
    let mut root_node = Node::new().with_name("root");
    root_node.children.push(elbow);
    let root = scene.add_node(root_node);
    scene.add_root(root);
    let skeleton = scene.add_skeleton(Skeleton {
        name: Some("arm".to_string()),
        joints: vec![root, elbow],
        inverse_bind_matrices: vec![
            identity(),
            [
                [1.0, 0.0, 0.0, -1.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        ],
    });
    let skin = scene.add_skin(Skin::new(skeleton).with_root(root));
    let mut mesh_node = Node::new().with_mesh(mesh);
    mesh_node.skin = Some(skin);
    let mesh_node = scene.add_node(mesh_node);
    scene.add_root(mesh_node);

    let anim = Animation::new(Some("bend".to_string()))
        .with_channel(
            elbow,
            AnimationProperty::Rotation,
            AnimationSampler {
                keyframes: vec![0.0, 1.0],
                values: AnimationValues::Quat(vec![
                    [0.0, 0.0, 0.0, 1.0],
                    quat_axis_angle([0.0, 0.0, 1.0], FRAC_PI_2),
                ]),
                interpolation: Interpolation::Linear,
            },
        )
        .with_channel(
            mesh_node,
            AnimationProperty::MorphWeights,
            AnimationSampler {
                keyframes: vec![0.0, 1.0],
                values: AnimationValues::Scalar(vec![0.0, 1.0]),
                interpolation: Interpolation::Linear,
            },
        );
    scene.add_animation(anim);
    add_light(
        &mut scene,
        Light::Directional {
            color: [1.0; 3],
            intensity: 3.0,
        },
        [0.0; 3],
        [0.0, 0.0, -1.0],
    );
    add_camera(&mut scene, [1.0, 0.5, 4.0], [1.0, 0.5, 0.0], 0.9);
    scene
}

fn identity() -> [[f32; 4]; 4] {
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

/// A lit, camera-facing quad whose material uses a constant
/// tangent-space normal map tilting the shading normal toward `+X`
/// (texel `(0.5 + tilt/2, 0.5, …)` encoded), lit by a directional
/// light from `+X`. With `tilt = 0` the map is neutral.
pub fn normal_mapped_quad(tilt: f32) -> Scene3D {
    let mut scene = Scene3D::new();
    let n = normalize([tilt, 0.0, 1.0]);
    let enc = |v: f32| ((v * 0.5 + 0.5) * 255.0).round() as u8;
    let texel = [enc(n[0]), enc(n[1]), enc(n[2]), 255];
    let tex = scene.add_texture(raw_texture(1, 1, &texel, Sampler::default_sampler()));
    let mut m = diffuse_material([0.8, 0.8, 0.8], 1.0);
    m.normal_texture = Some(TextureRef::new(tex));
    let mat = scene.add_material(m);
    let mut p = quad(
        [
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0],
        ],
        [1.0, 1.0],
    );
    p.material = Some(mat);
    add_mesh_node(&mut scene, p, Transform::identity());
    add_light(
        &mut scene,
        Light::Directional {
            color: [1.0; 3],
            intensity: 2.0,
        },
        [0.0; 3],
        [-1.0, 0.0, -0.3],
    );
    add_camera(&mut scene, [0.0, 0.0, 2.5], [0.0; 3], 0.9);
    scene
}

// ---------------------------------------------------------------------
// Metrics.
// ---------------------------------------------------------------------

/// Mean absolute per-channel difference (RGB only, `0..=255` scale).
/// Panics if the dimensions differ.
pub fn mean_abs_error(a: &RgbaImage, b: &RgbaImage) -> f64 {
    assert_eq!((a.width, a.height), (b.width, b.height), "size mismatch");
    let mut sum = 0u64;
    let mut n = 0u64;
    for (p, q) in a.pixels_rgba().zip(b.pixels_rgba()) {
        for k in 0..3 {
            sum += p[k].abs_diff(q[k]) as u64;
            n += 1;
        }
    }
    if n == 0 {
        0.0
    } else {
        sum as f64 / n as f64
    }
}

/// Peak signal-to-noise ratio in dB over RGB (`f64::INFINITY` for
/// identical images).
pub fn psnr(a: &RgbaImage, b: &RgbaImage) -> f64 {
    assert_eq!((a.width, a.height), (b.width, b.height), "size mismatch");
    let mut se = 0f64;
    let mut n = 0u64;
    for (p, q) in a.pixels_rgba().zip(b.pixels_rgba()) {
        for k in 0..3 {
            let d = p[k] as f64 - q[k] as f64;
            se += d * d;
            n += 1;
        }
    }
    if se == 0.0 || n == 0 {
        return f64::INFINITY;
    }
    10.0 * (255.0f64 * 255.0 / (se / n as f64)).log10()
}

/// Mean RGBA over the pixel rectangle `[x0, x1) × [y0, y1)` (clamped
/// to the image).
pub fn region_mean(img: &RgbaImage, x0: u32, y0: u32, x1: u32, y1: u32) -> [f64; 4] {
    let mut acc = [0f64; 4];
    let mut n = 0f64;
    for y in y0..y1.min(img.height) {
        for x in x0..x1.min(img.width) {
            let p = img.pixel(x, y).expect("in range");
            for k in 0..4 {
                acc[k] += p[k] as f64;
            }
            n += 1.0;
        }
    }
    if n > 0.0 {
        for v in &mut acc {
            *v /= n;
        }
    }
    acc
}

/// Standard deviation of Rec. 709 luma (on the byte values) over
/// `[x0, x1) × [y0, y1)`.
pub fn region_stddev(img: &RgbaImage, x0: u32, y0: u32, x1: u32, y1: u32) -> f64 {
    let mut v = Vec::new();
    for y in y0..y1.min(img.height) {
        for x in x0..x1.min(img.width) {
            let p = img.pixel(x, y).expect("in range");
            v.push(0.2126 * p[0] as f64 + 0.7152 * p[1] as f64 + 0.0722 * p[2] as f64);
        }
    }
    if v.is_empty() {
        return 0.0;
    }
    let m = v.iter().sum::<f64>() / v.len() as f64;
    (v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / v.len() as f64).sqrt()
}

/// Rec. 709 luma of an RGBA byte pixel.
pub fn luma(p: [u8; 4]) -> f64 {
    0.2126 * p[0] as f64 + 0.7152 * p[1] as f64 + 0.0722 * p[2] as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scenes_validate() {
        for (name, s) in [
            ("cornell", cornell_box()),
            ("spheres", sphere_grid(3, 2)),
            ("floor", checker_floor(Sampler::default_sampler())),
            ("quad", textured_quad(Sampler::default_sampler())),
            ("alpha", alpha_planes()),
            ("shadow", shadow_box()),
            ("beam", skinned_morph_beam()),
            ("normal", normal_mapped_quad(0.5)),
        ] {
            if let Err(e) = s.validate() {
                panic!("{name}: {e:?}");
            }
            assert_eq!(s.cameras.len(), 1, "{name}");
        }
    }

    #[test]
    fn look_rotation_points_minus_z_along_dir() {
        for d in [
            [0.0, 0.0, -1.0],
            [1.0, 0.0, 0.0],
            [0.3, -0.8, 0.2],
            [0.0, -1.0, 0.0],
        ] {
            let q = look_rotation(d);
            let m = Transform::Trs {
                translation: [0.0; 3],
                rotation: q,
                scale: [1.0; 3],
            }
            .to_matrix();
            let fwd = [-m[0][2], -m[1][2], -m[2][2]];
            let dn = normalize(d);
            for k in 0..3 {
                assert!((fwd[k] - dn[k]).abs() < 1e-4, "{d:?} -> {fwd:?}");
            }
        }
    }

    #[test]
    fn metrics_basics() {
        let a = RgbaImage::filled(4, 4, [10, 20, 30, 255]);
        let mut b = a.clone();
        assert_eq!(mean_abs_error(&a, &b), 0.0);
        assert!(psnr(&a, &b).is_infinite());
        b.set_pixel(0, 0, [13, 20, 30, 255]);
        assert!((mean_abs_error(&a, &b) - 3.0 / 48.0).abs() < 1e-9);
        assert!(psnr(&a, &b) > 40.0);
        assert_eq!(region_mean(&a, 0, 0, 2, 2)[1], 20.0);
        assert_eq!(region_stddev(&a, 0, 0, 4, 4), 0.0);
    }
}
