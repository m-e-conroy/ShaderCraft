
/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useRef, useEffect, useCallback } from 'react';
import { createRoot } from 'react-dom/client';
import { GoogleGenAI, Type } from "@google/genai";

declare var BABYLON: any;
declare var pep: any;
declare var CodeMirror: any;
declare var prettier: any;
declare var prettierPlugins: any;

const PBR_VERTEX_SHADER = `
precision highp float;

// Attributes
attribute vec3 position;
attribute vec2 uv;
attribute vec3 normal; // Vertex normal from the mesh

// Uniforms
uniform mat4 worldViewProjection;
uniform mat4 world; // World matrix for transforming normals and positions

// Varying
varying vec2 vUV;
varying vec3 vNormal; // Pass normal to fragment shader
varying vec3 vPositionW; // Pass world-space position to fragment shader

void main(void) {
    vec4 worldPosition = world * vec4(position, 1.0);
    gl_Position = worldViewProjection * vec4(position, 1.0);
    
    vUV = uv;
    vPositionW = worldPosition.xyz;

    // Transform normal to world space.
    mat3 normalMatrix = mat3(world);
    // Normalize the normal to ensure accurate lighting, especially if the mesh is scaled.
    vNormal = normalize(normalMatrix * normal);
}
`.trim();

const PBR_FRAGMENT_SHADER = `
precision highp float;

// Varying
varying vec2 vUV;
varying vec3 vNormal;
varying vec3 vPositionW;

// Uniforms
uniform float u_time;
uniform vec3 u_cameraPosition;

// Material Properties
uniform vec3 u_albedo;
uniform float u_metallic;
uniform float u_roughness;

// Lighting Uniforms
uniform vec3 u_lightColor;
uniform float u_lightIntensity;
uniform int u_lightType; // 0: directional, 1: point
uniform vec3 u_lightDirection;
uniform vec3 u_lightPosition;

// Environment/Reflection Uniforms
uniform samplerCube u_envTexture;
uniform int u_hasEnvTexture;

const float PI = 3.14159265359;

// PBR Functions
// 1. Normal Distribution Function (Trowbridge-Reitz GGX)
float DistributionGGX(vec3 N, vec3 H, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;
    float nom = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;
    return nom / denom;
}

// 2. Geometry Function (Schlick-GGX)
float GeometrySchlickGGX(float NdotV, float roughness) {
    float r = (roughness + 1.0);
    float k = (r * r) / 8.0;
    float nom = NdotV;
    float denom = NdotV * (1.0 - k) + k;
    return nom / denom;
}

// Smith's method for Geometry
float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2 = GeometrySchlickGGX(NdotV, roughness);
    float ggx1 = GeometrySchlickGGX(NdotL, roughness);
    return ggx1 * ggx2;
}

// 3. Fresnel Equation (Schlick's approximation)
vec3 fresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

void main(void) {
    // Input vectors
    vec3 N = normalize(vNormal);
    vec3 V = normalize(u_cameraPosition - vPositionW);
    
    vec3 lightDir;
    if (u_lightType == 0) { // Directional or Hemispheric
        lightDir = normalize(u_lightDirection);
    } else { // Point Light
        lightDir = normalize(u_lightPosition - vPositionW);
    }
    vec3 L = lightDir;
    vec3 H = normalize(V + L);

    // Base reflectivity at normal incidence (F0)
    vec3 F0 = vec3(0.04);
    F0 = mix(F0, u_albedo, u_metallic);

    // Direct lighting calculation (Cook-Torrance BRDF)
    float NDF = DistributionGGX(N, H, u_roughness);
    float G = GeometrySmith(N, V, L, u_roughness);
    vec3 F = fresnelSchlick(max(dot(H, V), 0.0), F0);

    vec3 kS = F;
    vec3 kD = vec3(1.0) - kS;
    kD *= (1.0 - u_metallic); // No diffuse for pure metals

    // Specular term
    vec3 numerator = NDF * G * F;
    float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.001;
    vec3 specular = numerator / denominator;

    // Additive direct light contribution
    float NdotL = max(dot(N, L), 0.0);
    vec3 directLighting = (kD * u_albedo / PI + specular) * u_lightColor * u_lightIntensity * NdotL;

    // Ambient lighting (a simple base)
    vec3 ambient = (vec3(0.05) * u_albedo) * (1.0 - u_metallic);
    
    vec3 litColor = directLighting + ambient;

    // --- Reflection Calculation ---
    // Blend environment map reflections on top of the lit surface.
    if (u_hasEnvTexture == 1) {
        vec3 viewDir = normalize(vPositionW - u_cameraPosition);
        vec3 reflectDir = reflect(viewDir, N);
        
        vec3 reflectionColor = textureCube(u_envTexture, reflectDir).rgb;

        // Use the Fresnel term to determine the strength of the reflection
        vec3 F_env = fresnelSchlick(max(dot(N, V), 0.0), F0);
        
        // A simple mix is not physically correct but gives good results.
        litColor = mix(litColor, reflectionColor, F_env);
    }

    gl_FragColor = vec4(litColor, 1.0);
}
`.trim();

const PHONG_FRAGMENT_SHADER = `
precision highp float;

varying vec2 vUV;
varying vec3 vNormal;
varying vec3 vPositionW;

uniform vec3 u_albedo;
uniform vec3 u_cameraPosition;
uniform vec3 u_lightColor;
uniform float u_lightIntensity;
uniform int u_lightType;
uniform vec3 u_lightDirection;
uniform vec3 u_lightPosition;

// Phong specific material properties
// We can repurpose PBR uniforms for this
// u_roughness can control shininess (lower roughness = higher shininess)
uniform float u_roughness; 

void main(void) {
    vec3 N = normalize(vNormal);
    vec3 V = normalize(u_cameraPosition - vPositionW);

    vec3 L;
    if (u_lightType == 0) { // Directional
        L = normalize(u_lightDirection);
    } else { // Point
        L = normalize(u_lightPosition - vPositionW);
    }

    // Ambient
    vec3 ambient = 0.1 * u_albedo;

    // Diffuse
    float NdotL = max(dot(N, L), 0.0);
    vec3 diffuse = u_albedo * NdotL * u_lightColor * u_lightIntensity;

    // Specular
    vec3 R = reflect(-L, N);
    float VdotR = max(dot(V, R), 0.0);
    float shininess = (1.0 - u_roughness) * 256.0;
    vec3 specular = u_lightColor * u_lightIntensity * pow(VdotR, shininess);

    vec3 finalColor = ambient + diffuse + specular;
    gl_FragColor = vec4(finalColor, 1.0);
}
`.trim();

const TOON_FRAGMENT_SHADER = `
precision highp float;

varying vec2 vUV;
varying vec3 vNormal;
varying vec3 vPositionW;

uniform vec3 u_albedo;
uniform vec3 u_cameraPosition;
uniform vec3 u_lightColor;
uniform float u_lightIntensity;
uniform int u_lightType;
uniform vec3 u_lightDirection;
uniform vec3 u_lightPosition;

// Toon shading steps
const int toonSteps = 3;

float celShade(float d) {
    float stepSize = 1.0 / float(toonSteps);
    return floor(d / stepSize) * stepSize;
}

void main(void) {
    vec3 N = normalize(vNormal);
    vec3 V = normalize(u_cameraPosition - vPositionW);

    vec3 L;
    if (u_lightType == 0) { // Directional
        L = normalize(u_lightDirection);
    } else { // Point
        L = normalize(u_lightPosition - vPositionW);
    }

    // Diffuse calculation
    float NdotL = max(dot(N, L), 0.0);
    
    // Apply cel shading
    float diffuseIntensity = celShade(NdotL);
    vec3 diffuse = u_albedo * diffuseIntensity * u_lightColor * u_lightIntensity;

    // Rim lighting (optional, for effect)
    float rim = 1.0 - max(dot(V, N), 0.0);
    rim = smoothstep(0.2, 0.5, rim);
    vec3 rimColor = vec3(1.0) * rim * 0.5;

    vec3 finalColor = diffuse + rimColor;

    gl_FragColor = vec4(finalColor, 1.0);
}
`.trim();

const UNLIT_FRAGMENT_SHADER = `
precision highp float;

varying vec2 vUV;

// We only need the albedo color for an unlit shader
uniform vec3 u_albedo;

// You could add a texture sampler here too
// uniform sampler2D u_texture;

void main(void) {
    // Just output the solid color
    gl_FragColor = vec4(u_albedo, 1.0);

    // Or, if using a texture:
    // gl_FragColor = texture2D(u_texture, vUV);
}
`.trim();


const SHADER_TEMPLATES = {
    'pbr': { name: 'Default PBR (Physical)', vertex: PBR_VERTEX_SHADER, fragment: PBR_FRAGMENT_SHADER },
    'phong': { name: 'Basic Phong (Classic)', vertex: PBR_VERTEX_SHADER, fragment: PHONG_FRAGMENT_SHADER },
    'toon': { name: 'Stylized Toon Shading', vertex: PBR_VERTEX_SHADER, fragment: TOON_FRAGMENT_SHADER },
    'unlit': { name: 'Unlit/Texture Passthrough', vertex: PBR_VERTEX_SHADER, fragment: UNLIT_FRAGMENT_SHADER },
};


const SHADER_SCHEMA = {
  type: Type.OBJECT,
  properties: {
    vertexShader: {
      type: Type.STRING,
      description: "The complete GLSL code for the vertex shader."
    },
    fragmentShader: {
      type: Type.STRING,
      description: "The complete GLSL code for the fragment shader."
    }
  },
  required: ["vertexShader", "fragmentShader"]
};

const SHADER_PRESETS = [
    { name: 'Matte Plastic', albedo: '#c73333', metallic: 0.0, roughness: 0.8 },
    { name: 'Polished Gold', albedo: '#ffd700', metallic: 1.0, roughness: 0.1 },
    { name: 'Rough Steel', albedo: '#b8b8b8', metallic: 1.0, roughness: 0.7 },
    { name: 'Shiny Porcelain', albedo: '#f0f0f0', metallic: 0.0, roughness: 0.2 },
    { name: 'Rubber Tire', albedo: '#202020', metallic: 0.0, roughness: 0.9 },
];

interface SavedShader {
    name: string;
    vertex: string;
    fragment: string;
    material?: {
        albedo: string;
        metallic: number;
        roughness: number;
    }
}

interface RefinementSelection {
    code: string;
    editor: 'vertex' | 'fragment';
}

/**
 * Robust manual GLSL formatter to ensure newlines and indentation.
 * Use this as a reliable fallback when external formatters fail.
 */
const manualFormatGlsl = (code: string): string => {
  if (!code) return '';
  
  // Clean string: remove any existing excessive whitespace if it seems like a one-liner
  const isOneLiner = !code.includes('\n') || code.split('\n').length < 5;
  let cleaned = code;
  if (isOneLiner) {
    cleaned = code.replace(/\s+/g, ' ').trim();
  }

  let formatted = '';
  let indent = 0;
  const step = '    '; // 4 space indentation
  
  for (let i = 0; i < cleaned.length; i++) {
    const char = cleaned[i];
    
    if (char === '{') {
      formatted = formatted.trimEnd();
      formatted += ' {\n' + step.repeat(++indent);
    } else if (char === '}') {
      indent = Math.max(0, indent - 1);
      formatted = formatted.trimEnd();
      formatted += '\n' + step.repeat(indent) + '}\n' + step.repeat(indent);
    } else if (char === ';') {
      // Basic check for 'for' loops to avoid splitting them
      const lastLine = formatted.split('\n').pop() || '';
      const openParens = (lastLine.match(/\(/g) || []).length;
      const closeParens = (lastLine.match(/\)/g) || []).length;
      
      if (openParens > closeParens) {
        formatted += '; ';
      } else {
        formatted += ';\n' + step.repeat(indent);
      }
    } else if (char === '\n' && !isOneLiner) {
      formatted += '\n' + step.repeat(indent);
    } else {
      // Prevent double spacing if we're working with cleaned text
      if (char === ' ' && formatted.endsWith(' ')) continue;
      formatted += char;
    }
  }
  
  return formatted
    .split('\n')
    .map(line => line.trimEnd())
    .join('\n')
    .replace(/\n\s*\n/g, '\n\n') // Normalize empty lines
    .trim();
};

// Function to safely parse JSON from localStorage
const getInitialState = (key: string, defaultValue: any): any => {
    try {
        const storedValue = localStorage.getItem(key);
        if (storedValue) {
            return JSON.parse(storedValue);
        }
    } catch (error) {
        console.error(`Failed to parse ${key} from local storage:`, error);
    }
    return defaultValue;
};

/**
 * Validates GLSL code by attempting to compile it in a temporary WebGL context.
 */
const glslValidator = (code: string, type: 'vertex' | 'fragment'): any[] => {
    const validator = glslValidator as any;
    if (!validator.gl) {
        const canvas = document.createElement('canvas');
        validator.gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
        if (!validator.gl) {
            console.warn("WebGL is not available for GLSL validation.");
            return [];
        }
    }
    const gl = validator.gl;

    const shader = gl.createShader(type === 'vertex' ? gl.VERTEX_SHADER : gl.FRAGMENT_SHADER);
    if (!shader) return [];

    gl.shaderSource(shader, code);
    gl.compileShader(shader);

    const errors: any[] = [];
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        const infoLog = gl.getShaderInfoLog(shader) || 'Unknown GLSL compilation error';
        const lines = infoLog.split('\n');
        
        for (const line of lines) {
            const match = line.match(/ERROR: 0:(\d+):(.*)/);
            if (match) {
                const lineNumber = parseInt(match[1], 10);
                const message = match[2].trim();
                if (lineNumber > 0) {
                     errors.push({
                        from: CodeMirror.Pos(lineNumber - 1, 0),
                        to: CodeMirror.Pos(lineNumber - 1, 1000),
                        message: message,
                        severity: 'error'
                    });
                }
            }
        }
    }
    
    gl.deleteShader(shader);
    return errors;
};

const PROMPT_CATEGORIES = {
    material: ['Metal', 'Wood', 'Stone', 'Fabric', 'Glass', 'Liquid', 'Plasma', 'Crystal', 'Hair', 'Fur', 'Skin', 'Slime', 'Energy', 'Bone', 'Ice', 'Coral', 'Paper', 'Leather', 'Ceramic', 'Meteorite', 'Damascus Steel', 'Petrified Wood', 'Obsidian', 'Geode', 'Nanoweave', 'Cracked Glass', 'Quicksilver', 'Ectoplasm'],
    finish: ['Polished', 'Rough', 'Matte', 'Anisotropic', 'Pearlescent', 'Wet', 'Dusty', 'Scratched', 'Corroded', 'Oily', 'Varnished', 'Crystalline', 'Sandy', 'Chipped', 'Embossed', 'Engraved'],
    effect: ['Glowing', 'Pulsating', 'Flowing', 'Shimmering', 'Cracking with light', 'Dissolving', 'Morphing', 'Breathing', 'Electric Arcs', 'Emitting Particles', 'Swirling', 'Bubbling', 'Freezing', 'Burning', 'Glitching', 'Pixelating'],
    pattern: ['Hexagonal tiles', 'Organic veins', 'Alien circuits', 'Wood grain', 'Fish scales', 'Fractal patterns', 'Hieroglyphics', 'Woven patterns', 'Honeycomb', 'Giger-esque biomechanical', 'Tessellated geometry', 'Camouflage', 'Paisley', 'Cracks', 'Craters', 'Topographical lines'],
    mood: ['Celestial', 'Abyssal', 'Volcanic', 'Botanical', 'Technological', 'Anatomical', 'Architectural', 'Ghostly', 'Glacial', 'Ancient', 'Futuristic', 'Magical', 'Corrupted', 'Dreamlike', 'Eldritch', 'Sacred', 'Industrial', 'Steampunk', 'Post-apocalyptic', 'Fairy-tale']
};

const generateRandomAiPrompt = (): string => {
    const getRandomItem = (arr: string[]): string => arr[Math.floor(Math.random() * arr.length)];
    const numCategories = 3 + Math.floor(Math.random() * 2);
    const selectedCategoryKeys = (Object.keys(PROMPT_CATEGORIES) as (keyof typeof PROMPT_CATEGORIES)[])
        .sort(() => 0.5 - Math.random())
        .slice(0, numCategories);
    const elements = selectedCategoryKeys.map(cat => getRandomItem(PROMPT_CATEGORIES[cat]));
    return `You are a creative art director. Your task is to take a set of keywords and expand them into a single, short, and highly imaginative prompt for a GLSL shader effect. Be descriptive and vivid.
    
Keywords: "${elements.join(', ')}"

Example Output for "Polished, Obsidian, Glowing": 'A polished obsidian stone surface with ethereal pastel veins cracking and glowing from within.'

Return ONLY the raw text for the final prompt. Do not include any extra words, formatting, or explanations.`;
};


const App = () => {
    const [vertexCode, setVertexCode] = useState<string>(PBR_VERTEX_SHADER);
    const [fragmentCode, setFragmentCode] = useState<string>(PBR_FRAGMENT_SHADER);
    const [prompt, setPrompt] = useState<string>('');
    const [isLoading, setIsLoading] = useState<boolean>(false);
    const [promptActionLoading, setPromptActionLoading] = useState<'random' | 'enhance' | null>(null);
    const [error, setError] = useState<string>('');
    const [activeTab, setActiveTab] = useState<'vertex' | 'fragment'>('fragment');
    const [selectedMesh, setSelectedMesh] = useState<string>('sphere');
    const [meshResolution, setMeshResolution] = useState<number>(32);
    const [showWireframe, setShowWireframe] = useState<boolean>(false);
    const [lightState, setLightState] = useState({
        type: 'hemispheric',
        intensity: 1.0,
        diffuse: '#ffffff',
        direction: { x: 1, y: 1, z: 0 }
    });
     const [materialState, setMaterialState] = useState({
        albedo: '#b3b3b3',
        metallic: 0.1,
        roughness: 0.5,
    });
    const [cameraState, setCameraState] = useState({
        type: 'arc',
        fov: 60,
        speed: 0.2,
        inertia: 0.8,
    });
    const [environmentTexture, setEnvironmentTexture] = useState<string | null>(null);
    const [liveReload, setLiveReload] = useState<boolean>(false);
    const [shaderName, setShaderName] = useState<string>('');
    const [savedShaders, setSavedShaders] = useState<SavedShader[]>(() => getInitialState('shadercraft_shaders', []));
    const [selectedShader, setSelectedShader] = useState<string>('');
    const [selectedTemplate, setSelectedTemplate] = useState<string>('');
    const [panelOrder, setPanelOrder] = useState<string[]>(() => getInitialState('shadercraft_panel_order', ['settings', 'ai', 'scene', 'project']));
    const [collapsedPanels, setCollapsedPanels] = useState<Record<string, boolean>>(() => getInitialState('shadercraft_collapsed_panels', {}));
    const [postProcessingState, setPostProcessingState] = useState({
        bloom: { enabled: false, threshold: 0.8, weight: 0.3, kernel: 64 },
        fxaa: { enabled: true },
        grain: { enabled: false, intensity: 10 },
        chromaticAberration: { enabled: false, aberrationAmount: 30 },
    });
    const [selectedPreset, setSelectedPreset] = useState<string>('');
    const [llmProvider, setLlmProvider] = useState<'gemini' | 'local' | 'lmstudio' | 'openai'>(() => getInitialState('shadercraft_llm_provider', 'gemini'));
    const [localLlmEndpoint, setLocalLlmEndpoint] = useState<string>(() => getInitialState('shadercraft_llm_endpoint', 'http://localhost:11434/api/generate'));
    const [localLlmModel, setLocalLlmModel] = useState<string>(() => getInitialState('shadercraft_llm_model', 'codellama'));
    const [localLlmStatus, setLocalLlmStatus] = useState<'unchecked' | 'connected' | 'error'>('unchecked');
    const [lmStudioUrl, setLmStudioUrl] = useState<string>(() => getInitialState('shadercraft_lmstudio_url', 'http://192.168.68.56:1234'));
    const [lmStudioStatus, setLmStudioStatus] = useState<'unchecked' | 'connected' | 'error'>('unchecked');
    const [selectedLmStudioModel, setSelectedLmStudioModel] = useState<string>(() => getInitialState('shadercraft_lmstudio_model', ''));
    const [lmStudioModels, setLmStudioModels] = useState<string[]>([]);
    const [isFetchingLmStudioModels, setIsFetchingLmStudioModels] = useState<boolean>(false);
    const [openAiUrl, setOpenAiUrl] = useState<string>(() => getInitialState('shadercraft_openai_url', 'http://localhost:8000/v1'));
    const [openAiModel, setOpenAiModel] = useState<string>(() => getInitialState('shadercraft_openai_model', ''));
    const [openAiStatus, setOpenAiStatus] = useState<'unchecked' | 'connected' | 'error'>('unchecked');
    const [openAiModels, setOpenAiModels] = useState<string[]>([]);
    const [isFetchingOpenAiModels, setIsFetchingOpenAiModels] = useState<boolean>(false);


    const [hasSelection, setHasSelection] = useState<boolean>(false);
    const [isRefining, setIsRefining] = useState<boolean>(false);
    const [refineModalOpen, setRefineModalOpen] = useState<boolean>(false);
    const [refinementPrompt, setRefinementPrompt] = useState<string>('');
    const [refinementSelection, setRefinementSelection] = useState<RefinementSelection | null>(null);

    const [timeState, setTimeState] = useState({ playing: true, time: 0.0 });
    const [confirmModalState, setConfirmModalState] = useState({
        isOpen: false,
        title: '',
        message: '',
        onConfirm: () => {},
    });

    const GEMINI_RATE_LIMIT_ERROR_MESSAGE = 'GEMINI_RATE_LIMIT_ERROR';
    const LMSTUDIO_CONNECTION_ERROR_MESSAGE = `LMSTUDIO_CONNECTION_ERROR`;
    const LMSTUDIO_INVALID_URL_ERROR_MESSAGE = `LMSTUDIO_INVALID_URL_ERROR`;
    const LOCAL_LLM_CONNECTION_ERROR_MESSAGE = "LOCAL_LLM_CONNECTION_ERROR";
    const OPENAI_CONNECTION_ERROR_MESSAGE = "OPENAI_CONNECTION_ERROR";
    const OPENAI_INVALID_URL_ERROR_MESSAGE = "OPENAI_INVALID_URL_ERROR";
    const TIMEOUT_ERROR_MESSAGE = "TIMEOUT_ERROR";


    const babylonCanvas = useRef<HTMLCanvasElement | null>(null);
    const sceneRef = useRef<any>(null);
    const engineRef = useRef<any>(null);
    const cameraRef = useRef<any>(null);
    const meshRef = useRef<any>(null);
    const lightRef = useRef<any>(null);
    const skyboxRef = useRef<any>(null);
    const ppPipelineRef = useRef<any>(null);
    const lightStateRef = useRef(lightState);
    const materialStateRef = useRef(materialState);
    const timeStateRef = useRef(timeState);
    const prevSelectedMeshRef = useRef<string | undefined>(undefined);

    const vertexEditorContainer = useRef<HTMLDivElement | null>(null);
    const fragmentEditorContainer = useRef<HTMLDivElement | null>(null);
    const vertexCmRef = useRef<any>(null);
    const fragmentCmRef = useRef<any>(null);

    const dragItem = useRef<number | null>(null);
    const dragOverItem = useRef<number | null>(null);

    useEffect(() => {
        lightStateRef.current = lightState;
    }, [lightState]);

    useEffect(() => {
        materialStateRef.current = materialState;
    }, [materialState]);

    useEffect(() => {
        timeStateRef.current = timeState;
    }, [timeState]);


    useEffect(() => {
        localStorage.setItem('shadercraft_llm_provider', JSON.stringify(llmProvider));
        localStorage.setItem('shadercraft_llm_endpoint', JSON.stringify(localLlmEndpoint));
        localStorage.setItem('shadercraft_llm_model', JSON.stringify(localLlmModel));
        localStorage.setItem('shadercraft_lmstudio_url', JSON.stringify(lmStudioUrl));
        localStorage.setItem('shadercraft_lmstudio_model', JSON.stringify(selectedLmStudioModel));
        localStorage.setItem('shadercraft_openai_url', JSON.stringify(openAiUrl));
        localStorage.setItem('shadercraft_openai_model', JSON.stringify(openAiModel));
    }, [llmProvider, localLlmEndpoint, localLlmModel, lmStudioUrl, selectedLmStudioModel, openAiUrl, openAiModel]);

    useEffect(() => {
        localStorage.setItem('shadercraft_panel_order', JSON.stringify(panelOrder));
    }, [panelOrder]);

    useEffect(() => {
        localStorage.setItem('shadercraft_collapsed_panels', JSON.stringify(collapsedPanels));
    }, [collapsedPanels]);

    useEffect(() => {
        if (llmProvider !== 'local' || !localLlmEndpoint) {
            setLocalLlmStatus('unchecked');
            return;
        };

        const controller = new AbortController();
        const timeoutId = setTimeout(async () => {
            try {
                const response = await fetch(localLlmEndpoint, {
                    method: 'HEAD',
                    signal: controller.signal,
                });
                if (response.ok || response.status === 405) {
                    setLocalLlmStatus('connected');
                } else {
                    setLocalLlmStatus('error');
                }
            } catch (err: any) {
                 if (err.name !== 'AbortError') {
                    setLocalLlmStatus('error');
                }
            }
        }, 500);

        return () => {
            clearTimeout(timeoutId);
            controller.abort();
        };
    }, [localLlmEndpoint, llmProvider]);

    const fetchOpenAiModels = useCallback(async () => {
        if (!openAiUrl || llmProvider !== 'openai') {
            setOpenAiStatus('unchecked');
            setOpenAiModels([]);
            return;
        }

        setIsFetchingOpenAiModels(true);
        setOpenAiStatus('unchecked');
        setError('');

        let modelsUrl;
        try {
            const url = new URL(openAiUrl);
            modelsUrl = new URL('models', `${url.href}${url.pathname.endsWith('/') ? '' : '/'}`).toString();
        } catch (e) {
            setError(OPENAI_INVALID_URL_ERROR_MESSAGE);
            setOpenAiStatus('error');
            setOpenAiModels([]);
            setIsFetchingOpenAiModels(false);
            return;
        }

        try {
            const response = await fetch(modelsUrl, {
                signal: AbortSignal.timeout(15000)
            });

            if (!response.ok) throw new Error(`Server responded with status: ${response.status}`);

            const data = await response.json();
            const models = data.data?.map((model: any) => model.id) || [];

            if (models.length === 0) throw new Error("No models found on the server.");

            setOpenAiModels(models);
            setOpenAiStatus('connected');

            const currentModel = getInitialState('shadercraft_openai_model', '');
            if (models.includes(currentModel)) {
                setOpenAiModel(currentModel);
            } else {
                setOpenAiModel(models[0]);
            }

        } catch (err: any) {
            console.error("OpenAI Compatible Server connection/fetch error:", err);
            if (err.name === 'TimeoutError' || (err instanceof DOMException && err.name === 'AbortError')) {
                setError(TIMEOUT_ERROR_MESSAGE);
            } else if (err instanceof TypeError && err.message === 'Failed to fetch') {
                setError(OPENAI_CONNECTION_ERROR_MESSAGE);
            } else {
                setError(err.message || 'An unknown error occurred while connecting to the server.');
            }
            setOpenAiStatus('error');
            setOpenAiModels([]);
        } finally {
            setIsFetchingOpenAiModels(false);
        }
    }, [openAiUrl, llmProvider]);


    const fetchLmStudioModels = useCallback(async () => {
        if (!lmStudioUrl || llmProvider !== 'lmstudio') {
            setLmStudioStatus('unchecked');
            setLmStudioModels([]);
            return;
        }
        
        setIsFetchingLmStudioModels(true);
        setLmStudioStatus('unchecked');
        setError('');
        
        let url;
        try {
            url = new URL('/v1/models', lmStudioUrl).toString();
        } catch (e) {
            setError(LMSTUDIO_INVALID_URL_ERROR_MESSAGE);
            setLmStudioStatus('error');
            setLmStudioModels([]);
            setIsFetchingLmStudioModels(false);
            return;
        }

        try {
            const response = await fetch(url, {
                signal: AbortSignal.timeout(15000)
            });
            
            if (!response.ok) throw new Error(`Server responded with status: ${response.status}`);

            const data = await response.json();
            const models = data.data?.map((model: any) => model.id) || [];
            
            if (models.length === 0) throw new Error("No models found on the server.");

            setLmStudioModels(models);
            setLmStudioStatus('connected');
            
            const currentModel = getInitialState('shadercraft_lmstudio_model', '');
            if (models.includes(currentModel)) {
                setSelectedLmStudioModel(currentModel);
            } else {
                setSelectedLmStudioModel(models[0]);
            }

        } catch (err: any) {
            console.error("LM Studio connection/fetch error:", err);
             if (err.name === 'TimeoutError' || (err instanceof DOMException && err.name === 'AbortError')) {
                setError(TIMEOUT_ERROR_MESSAGE);
            } else if (err instanceof TypeError && err.message === 'Failed to fetch') {
                setError(LMSTUDIO_CONNECTION_ERROR_MESSAGE);
            } else {
                setError(err.message || 'An unknown error occurred while connecting to LM Studio.');
            }
            setLmStudioStatus('error');
            setLmStudioModels([]);
        } finally {
            setIsFetchingLmStudioModels(false);
        }
    }, [lmStudioUrl, llmProvider]);

    useEffect(() => {
        if (llmProvider === 'lmstudio') {
            const timeoutId = setTimeout(() => {
                fetchLmStudioModels();
            }, 500);
            return () => clearTimeout(timeoutId);
        } else {
             setLmStudioStatus('unchecked');
             setLmStudioModels([]);
        }
    }, [lmStudioUrl, llmProvider, fetchLmStudioModels]);

    useEffect(() => {
        if (llmProvider === 'openai') {
            const timeoutId = setTimeout(() => {
                fetchOpenAiModels();
            }, 500);
            return () => clearTimeout(timeoutId);
        } else {
            setOpenAiStatus('unchecked');
            setOpenAiModels([]);
        }
    }, [openAiUrl, llmProvider, fetchOpenAiModels]);
    
    useEffect(() => {
        setError('');
    }, [llmProvider]);

    const showConfirmModal = (title: string, message: string, onConfirm: () => void) => {
        setConfirmModalState({ isOpen: true, title, message, onConfirm });
    };
    
    const closeConfirmModal = () => {
        setConfirmModalState({ isOpen: false, title: '', message: '', onConfirm: () => {} });
    };
    
    const handleConfirmAction = () => {
        confirmModalState.onConfirm();
        closeConfirmModal();
    };

    const handleClearShaders = () => {
        showConfirmModal(
            'Clear Shaders',
            'Are you sure you want to clear both shader editors? This action cannot be undone.',
            () => {
                setVertexCode('');
                setFragmentCode('');
                if (vertexCmRef.current) vertexCmRef.current.setValue('');
                if (fragmentCmRef.current) fragmentCmRef.current.setValue('');
                setSelectedShader('');
            }
        );
    };

    const handleSaveShader = () => {
        if (!shaderName.trim()) {
            alert("Please enter a name for your shader.");
            return;
        }

        const newShader: SavedShader = {
            name: shaderName.trim(),
            vertex: vertexCode,
            fragment: fragmentCode,
            material: { ...materialState }
        };
        
        const existingShaderIndex = savedShaders.findIndex(s => s.name === newShader.name);
        
        let updatedShaders;
        if (existingShaderIndex > -1) {
            updatedShaders = [...savedShaders];
            updatedShaders[existingShaderIndex] = newShader;
        } else {
            updatedShaders = [...savedShaders, newShader];
        }

        setSavedShaders(updatedShaders);
        localStorage.setItem('shadercraft_shaders', JSON.stringify(updatedShaders));
        setShaderName('');
        setSelectedShader(newShader.name);
    };

    const handleLoadShader = (name: string) => {
        setSelectedShader(name);
        if (!name) return;

        const shaderToLoad = savedShaders.find(s => s.name === name);
        if (shaderToLoad) {
            setVertexCode(shaderToLoad.vertex);
            setFragmentCode(shaderToLoad.fragment);
            if (shaderToLoad.material) {
                setMaterialState(shaderToLoad.material);
            } else {
                setMaterialState({ albedo: '#b3b3b3', metallic: 0.1, roughness: 0.5 });
            }
            setSelectedPreset('');
            setSelectedTemplate('');
        }
    };

    const handleDeleteShader = () => {
        if (!selectedShader) {
            alert("Please select a shader to delete.");
            return;
        }
        showConfirmModal(
            'Delete Shader',
            `Are you sure you want to delete the shader "${selectedShader}"?`,
            () => {
                const updatedShaders = savedShaders.filter(s => s.name !== selectedShader);
                setSavedShaders(updatedShaders);
                localStorage.setItem('shadercraft_shaders', JSON.stringify(updatedShaders));
                setSelectedShader('');
            }
        );
    };

    const handleExportShader = () => {
        if (!selectedShader) {
            alert("Please select a shader to export.");
            return;
        }

        const shaderToExport = savedShaders.find(s => s.name === selectedShader);
        if (!shaderToExport) {
            alert("Could not find the selected shader to export.");
            return;
        }

        const packageData = {
            name: shaderToExport.name,
            vertexShader: shaderToExport.vertex,
            fragmentShader: shaderToExport.fragment,
            material: shaderToExport.material
        };

        const jsonString = JSON.stringify(packageData, null, 2);
        const blob = new Blob([jsonString], { type: 'application/json' });
        const url = URL.createObjectURL(blob);

        const a = document.createElement('a');
        a.href = url;
        const filename = `${shaderToExport.name.toLowerCase().replace(/\s+/g, '-')}.json`;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    };

    const handleImportShader = (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (!file) return;

        const reader = new FileReader();
        reader.onload = (e) => {
            try {
                const text = e.target?.result as string;
                if (!text) throw new Error("File is empty.");
                const importedData = JSON.parse(text);
                if (!importedData.name || typeof importedData.name !== 'string' ||
                    !importedData.vertexShader || typeof importedData.vertexShader !== 'string' ||
                    !importedData.fragmentShader || typeof importedData.fragmentShader !== 'string') {
                    throw new Error("Invalid shader format. JSON must contain 'name', 'vertexShader', and 'fragmentShader' properties.");
                }

                const newShader: SavedShader = {
                    name: importedData.name.trim(),
                    vertex: importedData.vertexShader,
                    fragment: importedData.fragmentShader,
                    material: importedData.material
                };

                const existingShaderIndex = savedShaders.findIndex(s => s.name === newShader.name);
                
                const finishImport = (shaderToSave: SavedShader, isOverwrite: boolean) => {
                    let updatedShaders;
                    if (isOverwrite) {
                        const index = savedShaders.findIndex(s => s.name === shaderToSave.name);
                        updatedShaders = [...savedShaders];
                        updatedShaders[index] = shaderToSave;
                    } else {
                        updatedShaders = [...savedShaders, shaderToSave];
                    }

                    setSavedShaders(updatedShaders);
                    localStorage.setItem('shadercraft_shaders', JSON.stringify(updatedShaders));
                    
                    setSelectedShader(shaderToSave.name);
                    setVertexCode(shaderToSave.vertex);
                    setFragmentCode(shaderToSave.fragment);
                    if (shaderToSave.material) {
                        setMaterialState(shaderToSave.material);
                    }
                    
                    alert(`Shader "${shaderToSave.name}" ${isOverwrite ? 'overwritten' : 'imported'} successfully!`);
                };

                if (existingShaderIndex > -1) {
                    showConfirmModal(
                        'Overwrite Shader?',
                        `A shader named "${newShader.name}" already exists. Do you want to overwrite it?`,
                        () => finishImport(newShader, true)
                    );
                } else {
                    finishImport(newShader, false);
                }

            } catch (err: any) {
                console.error("Failed to import shader:", err);
                setError(err instanceof Error ? err.message : "An unknown error occurred during import.");
            } finally {
                event.target.value = '';
            }
        };
        reader.readAsText(file);
    };

    const handleTemplateChange = (templateKey: string) => {
        if (!templateKey) {
            setSelectedTemplate('');
            return;
        }

        showConfirmModal(
            'Load Shader Template?',
            'This will overwrite the current contents of the shader editors. Are you sure you want to continue?',
            () => {
                const template = SHADER_TEMPLATES[templateKey as keyof typeof SHADER_TEMPLATES];
                if (template) {
                    setVertexCode(template.vertex);
                    setFragmentCode(template.fragment);

                    switch (templateKey) {
                        case 'phong':
                            setMaterialState({ albedo: '#a0a0a0', metallic: 0.0, roughness: 0.3 });
                            break;
                        case 'toon':
                            setMaterialState({ albedo: '#4caf50', metallic: 0.0, roughness: 0.9 });
                            break;
                        case 'unlit':
                             setMaterialState({ albedo: '#007acc', metallic: 0.0, roughness: 1.0 });
                            break;
                        case 'pbr':
                        default:
                            setMaterialState({ albedo: '#b3b3b3', metallic: 0.1, roughness: 0.5 });
                            break;
                    }

                    setSelectedTemplate(templateKey);
                    setPrompt('');
                    setSelectedPreset('');
                    setSelectedShader('');
                }
            }
        );
    };

    const handleRunShader = useCallback(() => {
        if (!sceneRef.current || !meshRef.current) return;
        setError('');
        const scene = sceneRef.current;
        const existingMaterial = scene.getMaterialByName("customShader");
        if (existingMaterial) {
            existingMaterial.dispose();
        }
        const shaderMaterial = new BABYLON.ShaderMaterial(
            "customShader",
            scene,
            {
                vertexSource: vertexCode,
                fragmentSource: fragmentCode,
            },
            {
                attributes: ["position", "normal", "uv"],
                uniforms: [
                    "world", "worldView", "worldViewProjection", "view", "projection", 
                    "u_time", "u_lightColor", "u_lightIntensity", "u_lightDirection", 
                    "u_lightPosition", "u_lightType", "u_cameraPosition", "u_hasEnvTexture",
                    "u_albedo", "u_metallic", "u_roughness"
                ],
                samplers: ["u_envTexture"],
                onError: (sender: any, errors: string) => {
                    console.error("Shader Compilation Error:", errors);
                    setError(errors);
                },
            }
        );
        meshRef.current.material = shaderMaterial;
    }, [vertexCode, fragmentCode]);

    useEffect(() => {
        let animationFrameId: number;
        let lastTime = performance.now();
        const animate = (now: number) => {
            if (timeStateRef.current.playing) {
                const deltaTime = (now - lastTime) / 1000.0;
                setTimeState(prev => ({
                    ...prev,
                    time: prev.time + deltaTime
                }));
            }
            lastTime = now;
            animationFrameId = requestAnimationFrame(animate);
        };
        animationFrameId = requestAnimationFrame(animate);
        return () => {
            cancelAnimationFrame(animationFrameId);
        };
    }, []);

    useEffect(() => {
        if (!babylonCanvas.current) return;
        const engine = new BABYLON.Engine(babylonCanvas.current, true);
        engineRef.current = engine;
        const scene = new BABYLON.Scene(engine);
        sceneRef.current = scene;
        const defaultPipeline = new BABYLON.DefaultRenderingPipeline("defaultPipeline", true, scene, scene.cameras);
        ppPipelineRef.current = defaultPipeline;

        engine.runRenderLoop(() => {
            const material = scene.getMaterialByName("customShader");
            if (material && material.getClassName() === "ShaderMaterial") {
                const ls = lightStateRef.current;
                const ms = materialStateRef.current;
                const lightVector = new BABYLON.Vector3(ls.direction.x, ls.direction.y, ls.direction.z);
                (material as any).setFloat("u_time", timeStateRef.current.time);
                material.setColor3("u_albedo", BABYLON.Color3.FromHexString(ms.albedo));
                material.setFloat("u_metallic", ms.metallic);
                material.setFloat("u_roughness", ms.roughness);
                material.setFloat("u_lightIntensity", ls.intensity);
                material.setColor3("u_lightColor", BABYLON.Color3.FromHexString(ls.diffuse));
                if (ls.type === 'point') {
                    material.setInt("u_lightType", 1);
                    material.setVector3("u_lightPosition", lightVector);
                } else {
                    material.setInt("u_lightType", 0);
                    material.setVector3("u_lightDirection", lightVector);
                }
                if (scene.activeCamera) {
                    material.setVector3("u_cameraPosition", scene.activeCamera.position);
                }
                if (scene.environmentTexture && scene.environmentTexture.isReady()) {
                    material.setTexture("u_envTexture", scene.environmentTexture);
                    material.setInt("u_hasEnvTexture", 1);
                } else {
                    material.setInt("u_hasEnvTexture", 0);
                }
            }
            scene.render();
        });
        const resize = () => engine.resize();
        window.addEventListener('resize', resize);
        return () => {
            window.removeEventListener('resize', resize);
            ppPipelineRef.current?.dispose();
            engine.dispose();
        }
    }, []);

    useEffect(() => {
        if (!sceneRef.current || !babylonCanvas.current) return;
        const scene = sceneRef.current;
        const canvas = babylonCanvas.current;
        const currentCamera = cameraRef.current;
        const needsReplace = !currentCamera ||
            (cameraState.type === 'arc' && currentCamera.getClassName() !== 'ArcRotateCamera') ||
            (cameraState.type === 'free' && currentCamera.getClassName() !== 'FreeCamera');

        if (needsReplace) {
            let newCamera;
            const currentPosition = currentCamera?.position || new BABYLON.Vector3(0, 0, -5);
            const currentTarget = currentCamera?.getTarget ? currentCamera.getTarget() : BABYLON.Vector3.Zero();
            if (currentCamera && ppPipelineRef.current) {
                scene.postProcessRenderPipelineManager.detachCamerasFromRenderPipeline("defaultPipeline", currentCamera);
            }
            currentCamera?.dispose();
            if (cameraState.type === 'arc') {
                newCamera = new BABYLON.ArcRotateCamera("camera", -Math.PI / 2, Math.PI / 2.5, 5, BABYLON.Vector3.Zero(), scene);
                newCamera.setPosition(currentPosition);
                newCamera.setTarget(BABYLON.Vector3.Zero());
            } else {
                newCamera = new BABYLON.FreeCamera("camera", currentPosition, scene);
                newCamera.setTarget(currentTarget);
                newCamera.keysUp.push(87);
                newCamera.keysDown.push(83);
                newCamera.keysLeft.push(65);
                newCamera.keysRight.push(68);
                newCamera.keysUpward.push(69);
                newCamera.keysDownward.push(81);
            }
            newCamera.attachControl(canvas, true);
            cameraRef.current = newCamera;
            if (ppPipelineRef.current) {
                scene.postProcessRenderPipelineManager.attachCamerasToRenderPipeline("defaultPipeline", newCamera);
            }
        }
        const camera = cameraRef.current;
        if (camera) {
            camera.fov = cameraState.fov * (Math.PI / 180);
            if (camera.getClassName() === 'ArcRotateCamera') {
                camera.inertia = cameraState.inertia;
            } else if (camera.getClassName() === 'FreeCamera') {
                camera.speed = cameraState.speed;
            }
        }
    }, [cameraState]);
    
    useEffect(() => {
        if (!sceneRef.current) return;
        const scene = sceneRef.current;
        if (prevSelectedMeshRef.current !== selectedMesh || !meshRef.current) {
            if (meshRef.current) {
                meshRef.current.dispose();
            }
            let newMesh;
            switch (selectedMesh) {
                case 'cube':
                    newMesh = BABYLON.MeshBuilder.CreateBox("mesh", { size: 2 }, scene);
                    break;
                case 'torus':
                    newMesh = BABYLON.MeshBuilder.CreateTorus("mesh", { diameter: 3, thickness: 0.75, tessellation: meshResolution }, scene);
                    break;
                case 'plane':
                    newMesh = BABYLON.MeshBuilder.CreateGround("mesh", { width: 2.5, height: 2.5, subdivisions: meshResolution }, scene);
                    break;
                case 'cylinder':
                     newMesh = BABYLON.MeshBuilder.CreateCylinder("mesh", {height: 3, diameter: 1.5, tessellation: meshResolution }, scene);
                     break;
                case 'sphere':
                default:
                    newMesh = BABYLON.MeshBuilder.CreateSphere("mesh", { diameter: 2, segments: meshResolution }, scene);
                    break;
            }
            meshRef.current = newMesh;
        }
        handleRunShader();
        prevSelectedMeshRef.current = selectedMesh;
    }, [selectedMesh, meshResolution, handleRunShader]);

    useEffect(() => {
        if (meshRef.current && meshRef.current.material) {
            meshRef.current.material.wireframe = showWireframe;
        }
    }, [showWireframe, meshRef.current?.material]);

    useEffect(() => {
        if (!liveReload) return;
        const handler = setTimeout(() => {
            handleRunShader();
        }, 500);
        return () => {
            clearTimeout(handler);
        };
    }, [vertexCode, fragmentCode, liveReload, handleRunShader]);

    useEffect(() => {
        if (!sceneRef.current) return;
        const scene = sceneRef.current;
        const lightTypeMap: { [key: string]: string } = {
            'hemispheric': 'HemisphericLight',
            'directional': 'DirectionalLight',
            'point': 'PointLight'
        };
        const currentLightClassName = lightRef.current?.getClassName();
        const desiredLightClassName = lightTypeMap[lightState.type];
        if (!lightRef.current || currentLightClassName !== desiredLightClassName) {
            if (lightRef.current) {
                lightRef.current.dispose();
            }
            const lightName = "sceneLight";
            const lightVector = new BABYLON.Vector3(lightState.direction.x, lightState.direction.y, lightState.direction.z);
            switch(lightState.type) {
                case 'directional':
                    lightRef.current = new BABYLON.DirectionalLight(lightName, lightVector, scene);
                    break;
                case 'point':
                    lightRef.current = new BABYLON.PointLight(lightName, lightVector, scene);
                    break;
                case 'hemispheric':
                default:
                     lightRef.current = new BABYLON.HemisphericLight(lightName, lightVector, scene);
                     break;
            }
        }
        const light = lightRef.current;
        if (light) {
            light.diffuse = BABYLON.Color3.FromHexString(lightState.diffuse);
            const vector = new BABYLON.Vector3(lightState.direction.x, lightState.direction.y, lightState.direction.z);
            if (light.direction) light.direction = vector;
            if (light.position) light.position = vector;
        }
    }, [lightState]);

    useEffect(() => {
        if (!sceneRef.current) return;
        const scene = sceneRef.current;
        let createdSkybox: any = null;
        let createdTexture: any = null;
        let textureUrlToRevoke: string | null = null;
        if (environmentTexture) {
            createdTexture = new BABYLON.EquiRectangularCubeTexture(environmentTexture, scene, 512);
            scene.environmentTexture = createdTexture;
            createdSkybox = scene.createDefaultSkybox(createdTexture, true, 1000, 0.5);
            skyboxRef.current = createdSkybox;
            if (environmentTexture.startsWith('blob:')) {
                textureUrlToRevoke = environmentTexture;
            }
        } else {
            scene.environmentTexture = null;
            skyboxRef.current = null;
        }
        return () => {
            if (createdSkybox) createdSkybox.dispose();
            if (createdTexture) createdTexture.dispose();
            if (textureUrlToRevoke) URL.revokeObjectURL(textureUrlToRevoke);
        };
    }, [environmentTexture]);

    useEffect(() => {
        const pipeline = ppPipelineRef.current;
        if (!pipeline) return;
        pipeline.bloomEnabled = postProcessingState.bloom.enabled;
        if (pipeline.bloomEnabled) {
            pipeline.bloomThreshold = postProcessingState.bloom.threshold;
            pipeline.bloomWeight = postProcessingState.bloom.weight;
            pipeline.bloomKernel = postProcessingState.bloom.kernel;
        }
        pipeline.fxaaEnabled = postProcessingState.fxaa.enabled;
        pipeline.grainEnabled = postProcessingState.grain.enabled;
        if (pipeline.grainEnabled) {
            pipeline.grain.intensity = postProcessingState.grain.intensity;
            pipeline.grain.animated = true;
        }
        pipeline.chromaticAberrationEnabled = postProcessingState.chromaticAberration.enabled;
        if (pipeline.chromaticAberrationEnabled) {
            pipeline.chromaticAberration.aberrationAmount = postProcessingState.chromaticAberration.aberrationAmount;
            pipeline.chromaticAberration.radialIntensity = 1;
        }
    }, [postProcessingState]);


    useEffect(() => {
        const setupEditor = (
            container: HTMLElement | null, 
            value: string, 
            mode: string, 
            cmRef: React.MutableRefObject<any>, 
            setCode: (code: string) => void,
            shaderType: 'vertex' | 'fragment'
        ) => {
            if (container && !cmRef.current) {
                const cm = CodeMirror(container, {
                    value: value,
                    mode: mode,
                    theme: 'material-darker',
                    lineNumbers: true,
                    gutters: ["CodeMirror-linenumbers", "CodeMirror-lint-markers"],
                    lint: {
                        getAnnotations: (code: string) => glslValidator(code, shaderType),
                    },
                });
                cm.on('change', (instance: any) => {
                    setCode(instance.getValue());
                    instance.performLint();
                });
                cm.on('cursorActivity', (instance: any) => {
                    if (vertexCmRef.current?.somethingSelected() || fragmentCmRef.current?.somethingSelected()) {
                         setHasSelection(true);
                    } else {
                         setHasSelection(false);
                    }
                });
                cmRef.current = cm;
            }
        };
        setupEditor(vertexEditorContainer.current, vertexCode, 'x-shader/x-vertex', vertexCmRef, setVertexCode, 'vertex');
        setupEditor(fragmentEditorContainer.current, fragmentCode, 'x-shader/x-fragment', fragmentCmRef, setFragmentCode, 'fragment');
    }, []);

    useEffect(() => {
        if (vertexCmRef.current && vertexCmRef.current.getValue() !== vertexCode) {
            vertexCmRef.current.setValue(vertexCode);
        }
    }, [vertexCode]);
    
    useEffect(() => {
        if (fragmentCmRef.current && fragmentCmRef.current.getValue() !== fragmentCode) {
            fragmentCmRef.current.setValue(fragmentCode);
        }
    }, [fragmentCode]);

    useEffect(() => {
        setTimeout(() => {
            if (activeTab === 'vertex') vertexCmRef.current?.refresh();
            if (activeTab === 'fragment') fragmentCmRef.current?.refresh();
        }, 1);
    }, [activeTab]);
    
    const extractJsonFromString = (str: string): string | null => {
        const markdownMatch = str.match(/```json\s*([\s\S]*?)\s*```/);
        if (markdownMatch && markdownMatch[1]) {
            return markdownMatch[1].trim();
        }
        const firstBraceIndex = str.indexOf('{');
        const lastBraceIndex = str.lastIndexOf('}');
        if (firstBraceIndex !== -1 && lastBraceIndex > firstBraceIndex) {
            return str.substring(firstBraceIndex, lastBraceIndex + 1).trim();
        }
        return null;
    };

    const handleAiError = (error: any) => {
        console.error("AI Error:", error);
        const errorMessage = error instanceof Error ? error.message : String(error);
        if (error instanceof TypeError && (errorMessage.includes('Invalid URL') || errorMessage.includes('Failed to construct'))) {
            if (llmProvider === 'lmstudio') setError(LMSTUDIO_INVALID_URL_ERROR_MESSAGE);
            else if (llmProvider === 'openai') setError(OPENAI_INVALID_URL_ERROR_MESSAGE);
            return;
        }
        if (error.name === 'TimeoutError' || (error instanceof DOMException && error.name === 'AbortError')) {
            setError(TIMEOUT_ERROR_MESSAGE);
            return;
        }
        if (llmProvider === 'gemini' && (errorMessage.includes('429') || errorMessage.includes('RESOURCE_EXHAUSTED'))) {
            setError(GEMINI_RATE_LIMIT_ERROR_MESSAGE);
            return;
        }
        if (error instanceof TypeError && errorMessage === 'Failed to fetch') {
            if (llmProvider === 'lmstudio') setError(LMSTUDIO_CONNECTION_ERROR_MESSAGE);
            else if (llmProvider === 'local') setError(LOCAL_LLM_CONNECTION_ERROR_MESSAGE);
            else if (llmProvider === 'openai') setError(OPENAI_CONNECTION_ERROR_MESSAGE);
            else setError(`Connection failed: ${errorMessage}`);
            return;
        }
        setError(`An error occurred: ${errorMessage}`);
    };
    
    const formatGlslCode = (code: string) => {
        try {
            if (typeof prettier !== 'undefined' && typeof prettierPlugins !== 'undefined' && prettierPlugins.glsl) {
                return prettier.format(code, {
                    parser: 'glsl-parser',
                    plugins: [prettierPlugins.glsl],
                });
            } else {
                return manualFormatGlsl(code);
            }
        } catch (e) {
            console.warn('Formatting failed, using fallback:', e);
            return manualFormatGlsl(code);
        }
    };

    const handleGenerateShader = async () => {
        setIsLoading(true);
        setError('');

        const isRefinement = vertexCode !== PBR_VERTEX_SHADER || fragmentCode !== PBR_FRAGMENT_SHADER;

        const systemInstruction = `You are an expert in GLSL and Babylon.js. Create GLSL shaders that will run within a Babylon.js ShaderMaterial.

Please provide the complete GLSL code for both the vertex and fragment shaders.

- CRITICAL: Use proper indentation (4 spaces) and newlines for all generated code. 
- CRITICAL: DO NOT return the entire shader as a single line. This is mandatory.
- CRITICAL: Include descriptive comments (using //) on separate lines from code to prevent syntax errors.
- DO NOT include the \`#version\` directive.
- Ensure syntactically perfect GLSL code.
- Attributes: \`vec3 position\`, \`vec3 normal\`, \`vec2 uv\`.
- Varyings: \`vUV\` (\`vec2\`), \`vNormal\` (\`vec3\`), \`vPositionW\` (\`vec3\`).
- Uniforms: \`worldViewProjection\`, \`world\`, \`u_time\`, \`u_lightColor\`, \`u_lightIntensity\`, \`u_lightDirection\`, \`u_lightPosition\`, \`u_lightType\`, \`u_cameraPosition\`, \`u_albedo\`, \`u_metallic\`, \`u_roughness\`, \`u_envTexture\`, \`u_hasEnvTexture\`.

Return ONLY the code in a JSON object with keys "vertexShader" and "fragmentShader".`;

        const userContent = `The user wants a shader with this effect: "${prompt}"
${
  isRefinement
    ? `Current Vertex Shader:
\`\`\`glsl
${vertexCode}
\`\`\`
Current Fragment Shader:
\`\`\`glsl
${fragmentCode}
\`\`\`
`
    : ''
}`;

        try {
            let rawResponseText: string | null = null;
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 60000);

            if (llmProvider === 'gemini') {
                const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
                const response = await ai.models.generateContent({
                    model: "gemini-3-flash-preview",
                    contents: userContent,
                    config: {
                        systemInstruction: systemInstruction,
                        responseMimeType: "application/json",
                        responseSchema: SHADER_SCHEMA,
                    }
                });
                rawResponseText = response.text.trim();
            } else if (llmProvider === 'lmstudio') {
                const url = new URL('/v1/chat/completions', lmStudioUrl).toString();
                const response = await fetch(url, {
                    method: 'POST',
                    signal: controller.signal,
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        model: selectedLmStudioModel,
                        messages: [{ role: 'system', content: systemInstruction }, { role: 'user', content: userContent }],
                        stream: false,
                    })
                });
                const responseData = await response.json();
                rawResponseText = responseData.choices?.[0]?.message?.content;
            } else if (llmProvider === 'openai') {
                const url = new URL('chat/completions', `${openAiUrl}${openAiUrl.endsWith('/') ? '' : '/'}`).toString();
                const response = await fetch(url, {
                    method: 'POST',
                    signal: controller.signal,
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        model: openAiModel,
                        messages: [{ role: 'system', content: systemInstruction }, { role: 'user', content: userContent }],
                        stream: false,
                        response_format: { type: 'json_object' }
                    })
                });
                const responseData = await response.json();
                rawResponseText = responseData.choices?.[0]?.message?.content;
            } else {
                const response = await fetch(localLlmEndpoint, {
                    method: 'POST',
                    signal: controller.signal,
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        model: localLlmModel,
                        prompt: `${systemInstruction}\n\n${userContent}`,
                        stream: false,
                        format: 'json'
                    })
                });
                const responseData = await response.json();
                rawResponseText = responseData.response || responseData.content || JSON.stringify(responseData);
            }
            clearTimeout(timeoutId);
            if (!rawResponseText) throw new Error("AI response was empty.");
            const jsonString = extractJsonFromString(rawResponseText) || rawResponseText;
            const shaderData = JSON.parse(jsonString);
            if (shaderData.vertexShader && shaderData.fragmentShader) {
                setVertexCode(formatGlslCode(shaderData.vertexShader));
                setFragmentCode(formatGlslCode(shaderData.fragmentShader));
                setSelectedPreset('');
                setSelectedTemplate('');
                setMaterialState({ albedo: '#b3b3b3', metallic: 0.1, roughness: 0.5 });
            } else {
                setError("AI response was missing shader code.");
            }
        } catch (e: any) {
            handleAiError(e);
        } finally {
            setIsLoading(false);
        }
    };

    const handlePromptAction = async (action: 'random' | 'enhance') => {
        setPromptActionLoading(action);
        setError('');
        const content = action === 'random'
            ? generateRandomAiPrompt()
            : `Enhance this shader idea into a short vivid description: "${prompt}". Return raw text only.`;
        try {
            let resultText: string = '';
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 20000);
            if (llmProvider === 'gemini') {
                const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
                const response = await ai.models.generateContent({ model: "gemini-3-flash-preview", contents: content });
                resultText = response.text;
            } else {
                // ... same fallback logic if needed ...
            }
            clearTimeout(timeoutId);
            if (resultText) setPrompt(resultText.trim().replace(/['"]+/g, ''));
        } catch (e: any) {
            handleAiError(e);
        } finally {
            setPromptActionLoading(null);
        }
    };

    const toggleInspector = () => {
        if (sceneRef.current) {
            if (sceneRef.current.debugLayer.isVisible()) sceneRef.current.debugLayer.hide();
            else sceneRef.current.debugLayer.show({ embedMode: true });
        }
    };

    const handleOpenRefineModal = () => {
        const cm = activeTab === 'vertex' ? vertexCmRef.current : fragmentCmRef.current;
        if (cm && cm.somethingSelected()) {
            setRefinementSelection({ code: cm.getSelection(), editor: activeTab });
            setRefinementPrompt('');
            setRefineModalOpen(true);
        }
    };

    const closeRefineModal = () => {
        setRefineModalOpen(false);
        setRefinementSelection(null);
    };
    
    const handleFormatCode = () => {
        if (activeTab === 'vertex') {
            const formatted = formatGlslCode(vertexCode);
            setVertexCode(formatted);
            if (vertexCmRef.current) vertexCmRef.current.setValue(formatted);
        } else {
            const formatted = formatGlslCode(fragmentCode);
            setFragmentCode(formatted);
            if (fragmentCmRef.current) fragmentCmRef.current.setValue(formatted);
        }
    };

    const handleRefineCode = async () => {
        if (!refinementSelection || !refinementPrompt) return;
        setIsRefining(true);
        setError('');
        const fullShaderCode = refinementSelection.editor === 'vertex' ? vertexCode : fragmentCode;
        const systemInstruction = `You are an expert GLSL assistant. Rewrite the selected code snippet based on the instruction.
IMPORTANT: Return ONLY raw GLSL code snippet. Use proper newlines and indentation. No markdown fences.`;
        const userContent = `Instruction: "${refinementPrompt}"\nFull code for context: ${fullShaderCode}\nSelected snippet: ${refinementSelection.code}`;
        try {
            let refinedCode: string = '';
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 45000);

            if (llmProvider === 'gemini') {
                const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
                const response = await ai.models.generateContent({
                    model: "gemini-3-flash-preview",
                    contents: userContent,
                    config: { systemInstruction: systemInstruction }
                });
                refinedCode = response.text.trim();
            } else {
                // ... other provider logic ...
            }
            clearTimeout(timeoutId);

            if (refinedCode) {
              const cm = refinementSelection.editor === 'vertex' ? vertexCmRef.current : fragmentCmRef.current;
              if (cm) cm.replaceSelection(formatGlslCode(refinedCode));
              closeRefineModal();
            }
        } catch (e: any) {
            handleAiError(e);
        } finally {
            setIsRefining(false);
        }
    };
    
    const handleDragStart = (e: React.DragEvent<HTMLSpanElement>, position: number) => {
        dragItem.current = position;
        setTimeout(() => {
            const panel = (e.target as HTMLElement).closest('.collapsible-panel');
            panel?.classList.add('dragging');
        }, 0);
    };

    const handleDragEnter = (e: React.DragEvent<HTMLDivElement>, position: number) => {
        dragOverItem.current = position;
    };

    const handleDrop = () => {
        if (dragItem.current === null || dragOverItem.current === null || dragItem.current === dragOverItem.current) return;
        const newPanelOrder = [...panelOrder];
        const dragItemContent = newPanelOrder[dragItem.current];
        newPanelOrder.splice(dragItem.current, 1);
        newPanelOrder.splice(dragOverItem.current, 0, dragItemContent);
        setPanelOrder(newPanelOrder);
    };

    const handleDragEnd = () => {
        document.querySelectorAll('.collapsible-panel.dragging').forEach(el => el.classList.remove('dragging'));
        dragItem.current = null;
        dragOverItem.current = null;
    };
    
    const togglePanel = (key: string) => {
        setCollapsedPanels(prev => ({ ...prev, [key]: !prev[key] }));
    };

    const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (e) => setEnvironmentTexture(e.target?.result as string);
            reader.readAsDataURL(file);
        }
        event.target.value = '';
    };

    const handleRandomBackground = async () => {
        setError('');
        try {
            const randomImageUrl = `https://picsum.photos/2048/1024?random=${Date.now()}`;
            const response = await fetch(randomImageUrl);
            const imageBlob = await response.blob();
            const objectUrl = URL.createObjectURL(imageBlob);
            setEnvironmentTexture(objectUrl);
        } catch (err: any) {
            setError("Could not load background.");
        }
    };

    const clearEnvironment = () => setEnvironmentTexture(null);

    const handlePresetChange = (presetName: string) => {
        setSelectedPreset(presetName);
        if (!presetName) return;
        const preset = SHADER_PRESETS.find(p => p.name === presetName);
        if (preset) {
            setMaterialState({ albedo: preset.albedo, metallic: preset.metallic, roughness: preset.roughness });
            setSelectedShader('');
            setSelectedTemplate('');
        }
    };

    return (
        <div className="app-container">
            <header className="app-header">
                <h1>ShaderCraft AI</h1>
                <div className="header-controls">
                    <button onClick={toggleInspector} className="button-secondary header-icon-button" aria-label="Toggle Babylon.js Inspector">
                        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor">
                            <path d="M12 8c1.1 0 2 .9 2 2s-.9 2-2 2-2-.9-2-2 .9-2 2-2zm0 8c-2.21 0-4-1.79-4-4s1.79-4 4-4 4 1.79 4 4-1.79 4-4 4zm8.99-6.5c-1.25-3.44-4.5-6-8.49-6S4.26 6.06 3.01 9.5c-.31.85-.31 1.76 0 2.6.75 2.06 2.39 3.71 4.46 4.67.92.42 1.93.63 2.98.63s2.06-.21 2.98-.63c2.07-.96 3.71-2.61 4.46-4.67.31-.84.31-1.75 0-2.6zM12 18c-3.31 0-6-2.69-6-6s2.69-6 6-6 6 2.69 6 6-2.69 6-6 6z"/>
                        </svg>
                    </button>
                    <button onClick={handleRunShader} disabled={isLoading} className="header-icon-button" aria-label="Run and apply shader code">
                        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor">
                            <path d="M8 5v14l11-7z"/>
                        </svg>
                    </button>
                </div>
            </header>

            <main className="main-layout">
                <section className="panel controls-panel" aria-label="Controls">
                    {panelOrder.map((key, index) => {
                        let title: string;
                        let content: React.ReactNode;
                        switch (key) {
                            case 'settings':
                                title = 'Settings';
                                content = (
                                    <>
                                        <div className="form-group">
                                            <label htmlFor="llm-provider-select">AI Provider</label>
                                            <select id="llm-provider-select" value={llmProvider} onChange={(e) => setLlmProvider(e.target.value as any)}>
                                                <option value="gemini">Gemini API</option>
                                                <option value="lmstudio">LM Studio</option>
                                                <option value="openai">OpenAI Compatible</option>
                                                <option value="local">Local LLM (Ollama)</option>
                                            </select>
                                        </div>
                                        {error && <pre className="error-message-inline">{error}</pre>}
                                    </>
                                );
                                break;
                            case 'ai':
                                title = 'AI Controls';
                                content = (
                                    <>
                                        <div className="form-group">
                                            <label htmlFor="template-select">Templates</label>
                                            <select id="template-select" value={selectedTemplate} onChange={(e) => handleTemplateChange(e.target.value)}>
                                                <option value="">-- Templates --</option>
                                                {Object.entries(SHADER_TEMPLATES).map(([key, { name }]) => <option key={key} value={key}>{name}</option>)}
                                            </select>
                                        </div>
                                        <div className="form-group">
                                            <div className="prompt-label-group">
                                                <label htmlFor="prompt-input">Prompt</label>
                                                <div className="prompt-actions">
                                                    <button onClick={() => handlePromptAction('random')} disabled={!!promptActionLoading || isLoading} className="button-small">
                                                        {promptActionLoading === 'random' ? <span className="loader" /> : 'Random'}
                                                    </button>
                                                    <button onClick={() => handlePromptAction('enhance')} disabled={!prompt || !!promptActionLoading || isLoading} className="button-small">
                                                         {promptActionLoading === 'enhance' ? <span className="loader" /> : 'Enhance'}
                                                    </button>
                                                </div>
                                            </div>
                                            <textarea id="prompt-input" value={prompt} onChange={(e) => setPrompt(e.target.value)} placeholder="Describe your effect..." />
                                        </div>
                                        <button onClick={handleGenerateShader} disabled={isLoading || !prompt}>
                                            {isLoading ? <span className="loader" /> : 'Generate'}
                                        </button>
                                    </>
                                );
                                break;
                            case 'scene':
                                title = 'Scene';
                                content = (
                                    <>
                                        <div className="form-group">
                                            <label htmlFor="mesh-select">Mesh</label>
                                            <select id="mesh-select" value={selectedMesh} onChange={(e) => setSelectedMesh(e.target.value)}>
                                                <option value="sphere">Sphere</option>
                                                <option value="cube">Cube</option>
                                                <option value="torus">Torus</option>
                                                <option value="plane">Plane</option>
                                                <option value="cylinder">Cylinder</option>
                                            </select>
                                        </div>
                                        <div className="form-group form-group-row">
                                            <label htmlFor="material-albedo">Albedo</label>
                                            <input id="material-albedo" type="color" value={materialState.albedo} onChange={(e) => setMaterialState(prev => ({ ...prev, albedo: e.target.value }))} />
                                        </div>
                                    </>
                                );
                                break;
                            case 'project':
                                title = 'Project';
                                content = (
                                    <>
                                        <div className="form-group">
                                            <label htmlFor="shader-name-input">Save New / Rename</label>
                                            <div className="input-with-button">
                                                <input id="shader-name-input" type="text" value={shaderName} onChange={(e) => setShaderName(e.target.value)} placeholder="Shader Name" />
                                                <button onClick={handleSaveShader} disabled={!shaderName.trim()} title="Save current shader to library">Save</button>
                                            </div>
                                        </div>
                                        
                                        <div className="control-divider" />
                                        
                                        <div className="form-group">
                                            <label htmlFor="library-select">Shader Library</label>
                                            <select id="library-select" value={selectedShader} onChange={(e) => handleLoadShader(e.target.value)}>
                                                <option value="">-- Load Shader --</option>
                                                {savedShaders.map(s => <option key={s.name} value={s.name}>{s.name}</option>)}
                                            </select>
                                        </div>

                                        <div className="button-group">
                                            <button onClick={handleDeleteShader} disabled={!selectedShader} className="button-danger" title="Delete selected shader">Delete</button>
                                            <button onClick={handleExportShader} disabled={!selectedShader} className="button-secondary" title="Export selected shader as JSON">Export</button>
                                        </div>

                                        <div className="button-group">
                                            <label className="file-upload-button">
                                                Import JSON
                                                <input type="file" accept=".json" onChange={handleImportShader} style={{ display: 'none' }} />
                                            </label>
                                            <button onClick={handleClearShaders} className="button-secondary" title="Clear current editor">Clear</button>
                                        </div>
                                    </>
                                );
                                break;
                            default: return null;
                        }
                        const isCollapsed = !!collapsedPanels[key];
                        return (
                            <div key={key} className={`collapsible-panel ${!isCollapsed ? 'is-expanded' : ''}`} onDragEnter={(e) => handleDragEnter(e, index)} onDrop={handleDrop} onDragOver={(e) => e.preventDefault()}>
                                <h2 className={`panel-title collapsible ${isCollapsed ? 'is-collapsed' : 'is-expanded'}`} onClick={() => togglePanel(key)}>
                                    <span className="drag-handle" draggable onDragStart={(e) => { e.stopPropagation(); handleDragStart(e, index); }} onDragEnd={handleDragEnd} onClick={(e) => e.stopPropagation()}>
                                        <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor"><path d="M11 18c0 1.1-.9 2-2 2s-2-.9-2-2 .9-2 2-2 2 .9 2 2zm-2-8c-1.1 0-2 .9-2 2s.9 2 2 2 2-.9 2-2-.9-2-2-2zm0-6c-1.1 0-2 .9-2 2s.9 2 2 2 2-.9 2-2-.9-2-2-2zm6 4c1.1 0 2-.9 2-2s-.9-2-2-2-2 .9-2 2 .9 2 2 2zm0 2c-1.1 0-2 .9-2 2s.9 2 2 2 2-.9 2-2-.9-2-2-2zm0 6c-1.1 0-2 .9-2 2s.9 2 2 2 2-.9 2-2-.9-2-2-2z"/></svg>
                                    </span>
                                    <span className="title-text">{title}</span>
                                </h2>
                                <div className={`panel-content ${isCollapsed ? 'collapsed' : ''}`}>
                                    {content}
                                </div>
                            </div>
                        )
                    })}
                </section>

                <section className="panel viewport-panel">
                    <canvas id="babylon-canvas" ref={babylonCanvas} touch-action="none" />
                </section>

                <section className="panel editor-panel">
                    <div className="editor-header">
                        <div className="editor-tabs">
                            <button className={`tab-button ${activeTab === 'vertex' ? 'active' : ''}`} onClick={() => setActiveTab('vertex')}>Vertex</button>
                            <button className={`tab-button ${activeTab === 'fragment' ? 'active' : ''}`} onClick={() => setActiveTab('fragment')}>Fragment</button>
                        </div>
                        <div className="editor-actions">
                             <button onClick={handleFormatCode} className="button-format">Format Code</button>
                        </div>
                    </div>
                    <div className="editor-content">
                        <div ref={vertexEditorContainer} style={{ display: activeTab === 'vertex' ? 'block' : 'none', height: '100%' }} />
                        <div ref={fragmentEditorContainer} style={{ display: activeTab === 'fragment' ? 'block' : 'none', height: '100%' }} />
                    </div>
                </section>
            </main>
            <footer className="app-footer">
                {isLoading || isRefining || promptActionLoading ? 'Processing...' : error ? 'Error occurred.' : 'Ready'}
            </footer>

            {refineModalOpen && (
                <div className="refine-modal-backdrop" onClick={closeRefineModal}>
                    <div className="refine-modal" onClick={(e) => e.stopPropagation()}>
                        <h3>Refine Code with AI</h3>
                        <div className="refine-modal-content">
                            <div className="form-group">
                                <label>Selection:</label>
                                <pre className="code-snippet"><code>{refinementSelection?.code}</code></pre>
                            </div>
                            <textarea
                                value={refinementPrompt}
                                onChange={(e) => setRefinementPrompt(e.target.value)}
                                placeholder="How to change it?"
                            />
                        </div>
                        <div className="refine-modal-actions">
                            <button onClick={closeRefineModal} className="button-secondary">Cancel</button>
                            <button onClick={handleRefineCode} disabled={!refinementPrompt || isRefining}>Refine</button>
                        </div>
                    </div>
                </div>
            )}

            {confirmModalState.isOpen && (
                <div className="confirm-modal-backdrop" onClick={closeConfirmModal}>
                    <div className="confirm-modal" onClick={(e) => e.stopPropagation()}>
                        <h3>{confirmModalState.title}</h3>
                        <p>{confirmModalState.message}</p>
                        <div className="confirm-modal-actions">
                            <button onClick={closeConfirmModal} className="button-secondary">Cancel</button>
                            <button onClick={handleConfirmAction} className="button-danger">Confirm</button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

const container = document.getElementById('root');
const root = createRoot(container!);
root.render(<App />);
