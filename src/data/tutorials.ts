export interface Lesson {
  id: string;
  phase: string;
  title: string;
  description: string;
  readingLinks: { title: string; url: string }[];
  starterVertex: string;
  starterFragment: string;
  checkpoints: string[];
  tutorContext: string;
}

export const tutorials: Lesson[] = [
  {
    id: 'phase1-intro',
    phase: 'Phase 1: Foundations',
    title: 'Introduction to Path Tracing & Rays',
    description: 'Understand how path tracing works conceptually before writing a single line of WebGL. This phase focuses on the theory of rays, bounces, and importance sampling.',
    readingLinks: [
      { title: 'Ray Tracing in One Weekend', url: 'https://raytracing.github.io/books/RayTracingInOneWeekend.html' }
    ],
    starterVertex: `precision highp float;
attribute vec3 position;
attribute vec2 uv;
uniform mat4 worldViewProjection;
varying vec2 vUV;
void main(void) {
    gl_Position = worldViewProjection * vec4(position, 1.0);
    vUV = uv;
}`,
    starterFragment: `precision highp float;
varying vec2 vUV;
void main(void) {
    // A simple color to start
    gl_FragColor = vec4(vUV.x, vUV.y, 0.5, 1.0);
}`,
    checkpoints: [
      'What is a ray? What is a bounce?',
      'What does "importance sampling" mean?',
      'Why does a path-traced image look noisy at first and then converge?'
    ],
    tutorContext: 'The user is learning the conceptual foundations of path tracing. Do not write code for them yet. Answer questions about rays, bounces, Monte Carlo integration, and importance sampling.'
  },
  {
    id: 'phase2-glsl-basics',
    phase: 'Phase 2: GLSL Shader Programming',
    title: 'Hello Fragment (gl_FragColor)',
    description: 'The best hands-on intro to GLSL. Learn how to output colors to the screen using the fragment shader.',
    readingLinks: [
      { title: 'The Book of Shaders', url: 'https://thebookofshaders.com' },
      { title: 'MDN GLSL Shaders Reference', url: 'https://developer.mozilla.org/en-US/docs/Games/Techniques/3D_on_the_web/GLSL_Shaders' }
    ],
    starterVertex: `precision highp float;
attribute vec3 position;
attribute vec2 uv;
uniform mat4 worldViewProjection;
varying vec2 vUV;
void main(void) {
    gl_Position = worldViewProjection * vec4(position, 1.0);
    vUV = uv;
}`,
    starterFragment: `precision highp float;
varying vec2 vUV;
uniform float u_time;

void main(void) {
    // TODO: Change this color to solid red!
    // Hint: vec4(Red, Green, Blue, Alpha)
    gl_FragColor = vec4(1.0, 1.0, 1.0, 1.0);
}`,
    checkpoints: [
      'Write a custom Fragment Shader that outputs solid red.',
      'Understand the difference between attribute, uniform, and varying.',
      'Use the vUV varying to create a gradient.'
    ],
    tutorContext: 'The user is learning basic GLSL. Guide them to change gl_FragColor to vec4(1.0, 0.0, 0.0, 1.0) for red. Explain uniforms and varyings if asked.'
  },
  {
    id: 'phase2-uniforms',
    phase: 'Phase 2: GLSL Shader Programming',
    title: 'Uniforms & Time',
    description: 'Learn how to pass data from JavaScript to your shader using Uniforms. We will use the u_time uniform to animate our shader.',
    readingLinks: [
      { title: 'The Book of Shaders: Uniforms', url: 'https://thebookofshaders.com/03/' }
    ],
    starterVertex: `precision highp float;
attribute vec3 position;
attribute vec2 uv;
uniform mat4 worldViewProjection;
varying vec2 vUV;
void main(void) {
    gl_Position = worldViewProjection * vec4(position, 1.0);
    vUV = uv;
}`,
    starterFragment: `precision highp float;
varying vec2 vUV;
// u_time is passed in from the app automatically!
uniform float u_time;

void main(void) {
    // TODO: Use the sin() function with u_time to make the color pulse
    float pulse = 1.0; 
    
    gl_FragColor = vec4(pulse, 0.0, 0.0, 1.0);
}`,
    checkpoints: [
      'Use the sin() function to create a pulsing value.',
      'Apply the pulsing value to the red channel of gl_FragColor.',
      'Experiment with multiplying u_time to change the pulse speed.'
    ],
    tutorContext: 'The user is learning about uniforms, specifically u_time. Guide them to use sin(u_time) to create a pulsing effect. If they ask about speed, suggest sin(u_time * 2.0).'
  },
  {
    id: 'phase3-custom-materials',
    phase: 'Phase 3: Custom Materials',
    title: 'Lighting & Normals',
    description: 'Understand how to use vertex normals to calculate basic lighting.',
    readingLinks: [
      { title: 'Three.js Journey - Shaders', url: 'https://threejs-journey.com/lessons/shaders' }
    ],
    starterVertex: `precision highp float;
attribute vec3 position;
attribute vec2 uv;
attribute vec3 normal;

uniform mat4 worldViewProjection;
uniform mat4 world;

varying vec2 vUV;
varying vec3 vNormal;

void main(void) {
    gl_Position = worldViewProjection * vec4(position, 1.0);
    vUV = uv;
    
    // Transform normal to world space
    mat3 normalMatrix = mat3(world);
    vNormal = normalize(normalMatrix * normal);
}`,
    starterFragment: `precision highp float;
varying vec2 vUV;
varying vec3 vNormal;

uniform vec3 u_lightDirection;

void main(void) {
    vec3 N = normalize(vNormal);
    vec3 L = normalize(u_lightDirection);
    
    // TODO: Calculate the dot product between the Normal and Light Direction
    // Hint: Use max(dot(N, L), 0.0) to prevent negative light
    float diffuse = 1.0;
    
    vec3 color = vec3(1.0, 0.5, 0.2) * diffuse;
    gl_FragColor = vec4(color, 1.0);
}`,
    checkpoints: [
      'Pass the normal attribute from the vertex to the fragment shader.',
      'Calculate the dot product of the normal and light direction.',
      'Apply the diffuse lighting to the base color.'
    ],
    tutorContext: 'The user is learning about normals and diffuse lighting. Guide them to use max(dot(N, L), 0.0) to calculate the diffuse intensity.'
  },
  {
    id: 'phase5-openpbr',
    phase: 'Phase 5: OpenPBR Integration',
    title: 'OpenPBR Concepts',
    description: 'Learn about the OpenPBR specification and how it standardizes physically based rendering materials.',
    readingLinks: [
      { title: 'OpenPBR Specification', url: 'https://openpbr.org' },
      { title: 'Adobe OpenPBR BSDF', url: 'https://github.com/adobe/openpbr-bsdf' }
    ],
    starterVertex: `precision highp float;
attribute vec3 position;
attribute vec2 uv;
attribute vec3 normal;
uniform mat4 worldViewProjection;
uniform mat4 world;
varying vec2 vUV;
varying vec3 vNormal;
varying vec3 vPositionW;
void main(void) {
    vec4 worldPosition = world * vec4(position, 1.0);
    gl_Position = worldViewProjection * vec4(position, 1.0);
    vUV = uv;
    vPositionW = worldPosition.xyz;
    mat3 normalMatrix = mat3(world);
    vNormal = normalize(normalMatrix * normal);
}`,
    starterFragment: `precision highp float;
varying vec2 vUV;
varying vec3 vNormal;
varying vec3 vPositionW;
uniform vec3 u_cameraPosition;
uniform vec3 u_albedo;
uniform float u_metallic;
uniform float u_roughness;

void main(void) {
    // In a full OpenPBR implementation, you would populate an OpenPBR_ResolvedInputs struct here.
    // For now, experiment with the basic PBR uniforms provided by the app.
    
    vec3 N = normalize(vNormal);
    vec3 V = normalize(u_cameraPosition - vPositionW);
    
    // Simple visualization of the view angle (Fresnel precursor)
    float NdotV = max(dot(N, V), 0.0);
    
    gl_FragColor = vec4(vec3(NdotV) * u_albedo, 1.0);
}`,
    checkpoints: [
      'Read the OpenPBR specification overview.',
      'Understand the difference between base color, metallic, and roughness.',
      'Visualize the NdotV (Normal dot View) value.'
    ],
    tutorContext: 'The user is learning about OpenPBR concepts. Explain how parameters like base color, metallic, and roughness interact in a physically based rendering model.'
  }
];
