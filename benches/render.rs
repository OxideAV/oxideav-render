//! Criterion benchmarks for the two live render backends.
//!
//! Scenes are synthesised procedurally (UV sphere + optional mirror
//! floor) so no fixture files are committed and every input is
//! reproducible from this source. Numbers land in `BENCHMARKS.md`.
//!
//! Scenario map:
//!
//! - **scanline_phong_960tri_256** / **raycast_phong_960tri_256** —
//!   the head-to-head: identical sphere scene, identical options,
//!   Phong shading at 256×256. The raycast row includes per-render
//!   scene baking + BVH build + per-pixel shadow rays.
//! - **scanline_flat_960tri_256** / **raycast_flat_960tri_256** —
//!   same head-to-head without lighting (Flat is unlit in both).
//! - **raycast_phong_3968tri_256** — triangle-count scaling on the
//!   BVH walk (~4× the triangles of the 960-triangle sphere).
//! - **raycast_mirror_floor_256** — Whitted recursion: metallic
//!   floor under the sphere doubles the ray tree for floor pixels.
//! - **scanline_phong_960tri_aa4_128** / **raycast_phong_960tri_aa4_128**
//!   — SSAA 4×: renders 512×512 samples for a 128×128 output.
//! - **raycast_bake_only_3968tri_1** — 1×1 output isolates scene
//!   bake + BVH build from per-pixel tracing.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use oxideav_mesh3d::{
    Indices, Material, MaterialId, Mesh, MeshId, Node, NodeId, Primitive, Scene3D, Topology,
};
use oxideav_render::{make_renderer, BackgroundColor, RenderBackend, RenderOptions, ShadingMode};

/// UV sphere: `segments × rings` quads, two triangles each, indexed,
/// with exact unit normals. `segments = 32, rings = 16` → 960
/// triangles (poles collapse one triangle per quad).
fn uv_sphere(segments: u32, rings: u32) -> Primitive {
    let mut prim = Primitive::new(Topology::Triangles);
    let mut normals = Vec::new();
    for r in 0..=rings {
        let theta = std::f32::consts::PI * (r as f32) / (rings as f32);
        for s in 0..=segments {
            let phi = 2.0 * std::f32::consts::PI * (s as f32) / (segments as f32);
            let n = [
                theta.sin() * phi.cos(),
                theta.cos(),
                theta.sin() * phi.sin(),
            ];
            prim.positions.push(n);
            normals.push(n);
        }
    }
    prim.normals = Some(normals);
    let stride = segments + 1;
    let mut idx = Vec::new();
    for r in 0..rings {
        for s in 0..segments {
            let a = r * stride + s;
            let b = a + 1;
            let c = a + stride;
            let d = c + 1;
            idx.extend_from_slice(&[a, c, b, b, c, d]);
        }
    }
    prim.indices = Some(Indices::U32(idx));
    prim
}

fn sphere_scene(segments: u32, rings: u32) -> Scene3D {
    let mut scene = Scene3D::new();
    scene
        .meshes
        .push(Mesh::new("sphere".to_string()).with_primitive(uv_sphere(segments, rings)));
    scene.nodes.push(Node {
        mesh: Some(MeshId(0)),
        ..Node::default()
    });
    scene.roots.push(NodeId(0));
    scene
}

/// Sphere resting on a metallic mirror floor — exercises the Whitted
/// reflection recursion for every floor pixel.
fn mirror_floor_scene() -> Scene3D {
    let mut scene = sphere_scene(32, 16);
    let mut floor = Primitive::new(Topology::Triangles);
    floor.positions = vec![
        [-4.0, -1.0, -4.0],
        [4.0, -1.0, -4.0],
        [-4.0, -1.0, 4.0],
        [4.0, -1.0, -4.0],
        [4.0, -1.0, 4.0],
        [-4.0, -1.0, 4.0],
    ];
    floor.normals = Some(vec![[0.0, 1.0, 0.0]; 6]);
    floor.material = Some(MaterialId(0));
    scene.materials.push(Material {
        base_color: [0.9, 0.9, 0.9, 1.0],
        metallic: 1.0,
        roughness: 0.0,
        ..Material::new()
    });
    let mesh_id = scene.meshes.len() as u32;
    scene
        .meshes
        .push(Mesh::new("floor".to_string()).with_primitive(floor));
    let node_id = scene.nodes.len() as u32;
    scene.nodes.push(Node {
        mesh: Some(MeshId(mesh_id)),
        ..Node::default()
    });
    scene.roots.push(NodeId(node_id));
    scene
}

fn opts(size: u32, shading: ShadingMode, aa: u32) -> RenderOptions {
    RenderOptions {
        width: size,
        height: size,
        shading,
        aa,
        background: BackgroundColor([16, 16, 24, 255]),
        ..RenderOptions::default()
    }
}

fn bench_backend(
    c: &mut Criterion,
    name: &str,
    backend: RenderBackend,
    scene: &Scene3D,
    options: &RenderOptions,
) {
    let mut renderer = make_renderer(backend).expect("backend must construct");
    c.bench_function(name, |b| {
        b.iter(|| {
            let img = renderer
                .render(black_box(scene), black_box(options))
                .expect("render");
            black_box(img)
        })
    });
}

fn benches(c: &mut Criterion) {
    let sphere = sphere_scene(32, 16); // 960 triangles
    let sphere4k = sphere_scene(64, 32); // 3968 triangles
    let mirror = mirror_floor_scene();

    let phong256 = opts(256, ShadingMode::Phong, 1);
    let flat256 = opts(256, ShadingMode::Flat, 1);
    let phong128aa4 = opts(128, ShadingMode::Phong, 4);
    let phong1 = opts(1, ShadingMode::Phong, 1);

    bench_backend(
        c,
        "scanline_phong_960tri_256",
        RenderBackend::Scanline,
        &sphere,
        &phong256,
    );
    bench_backend(
        c,
        "raycast_phong_960tri_256",
        RenderBackend::Raycast,
        &sphere,
        &phong256,
    );
    bench_backend(
        c,
        "scanline_flat_960tri_256",
        RenderBackend::Scanline,
        &sphere,
        &flat256,
    );
    bench_backend(
        c,
        "raycast_flat_960tri_256",
        RenderBackend::Raycast,
        &sphere,
        &flat256,
    );
    bench_backend(
        c,
        "raycast_phong_3968tri_256",
        RenderBackend::Raycast,
        &sphere4k,
        &phong256,
    );
    bench_backend(
        c,
        "raycast_mirror_floor_256",
        RenderBackend::Raycast,
        &mirror,
        &phong256,
    );
    bench_backend(
        c,
        "scanline_phong_960tri_aa4_128",
        RenderBackend::Scanline,
        &sphere,
        &phong128aa4,
    );
    bench_backend(
        c,
        "raycast_phong_960tri_aa4_128",
        RenderBackend::Raycast,
        &sphere,
        &phong128aa4,
    );
    bench_backend(
        c,
        "raycast_bake_only_3968tri_1",
        RenderBackend::Raycast,
        &sphere4k,
        &phong1,
    );
}

criterion_group!(render_benches, benches);
criterion_main!(render_benches);
