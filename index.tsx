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

const DEFAULT_VERTEX_SHADER = `
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

const DEFAULT_FRAGMENT_SHADER = `
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

// Function to safely parse JSON from localStorage
// FIX: Removed generics and used `any` to prevent parser errors from buggy linters.
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
 * This provides the most accurate, browser-specific syntax checking.
 * @param code The GLSL code string to validate.
 * @param type The type of shader, either 'vertex' or 'fragment'.
 * @returns An array of error objects formatted for the CodeMirror lint addon.
 */
const glslValidator = (code: string, type: 'vertex' | 'fragment'): any[] => {
    // Use a static canvas/context to avoid creating them repeatedly, improving performance.
    const validator = glslValidator as any;
    if (!validator.gl) {
        const canvas = document.createElement('canvas');
        // Try to get a WebGL2 context first for better feature support, fallback to WebGL1.
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
        const infoLog = gl.getShaderInfoLog() || 'Unknown GLSL compilation error';
        const lines = infoLog.split('\n');
        
        for (const line of lines) {
            // Standard error format: "ERROR: 0:<line>: <message>"
            const match = line.match(/ERROR: 0:(\d+):(.*)/);
            if (match) {
                const lineNumber = parseInt(match[1], 10);
                const message = match[2].trim();
                if (lineNumber > 0) {
                     errors.push({
                        from: CodeMirror.Pos(lineNumber - 1, 0),
                        to: CodeMirror.Pos(lineNumber - 1, 1000), // Highlight the whole line
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

// FIX: Removed `: React.FC` to simplify the component definition and avoid potential complex type-checking issues.
const App = () => {
    const [vertexCode, setVertexCode] = useState<string>(DEFAULT_VERTEX_SHADER);
    const [fragmentCode, setFragmentCode] = useState<string>(DEFAULT_FRAGMENT_SHADER);
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
        direction: { x: 1, y: 1, z: 0 } // Re-used for position in point lights
    });
     const [materialState, setMaterialState] = useState({
        albedo: '#b3b3b3',
        metallic: 0.1,
        roughness: 0.5,
    });
    const [environmentTexture, setEnvironmentTexture] = useState<string | null>(null);
    const [liveReload, setLiveReload] = useState<boolean>(false);
    const [shaderName, setShaderName] = useState<string>('');
    const [savedShaders, setSavedShaders] = useState<SavedShader[]>([]);
    const [selectedShader, setSelectedShader] = useState<string>('');
    const [panelOrder, setPanelOrder] = useState<string[]>(['ai', 'scene', 'project']);
    const [collapsedPanels, setCollapsedPanels] = useState<Record<string, boolean>>({});
    const [postProcessingState, setPostProcessingState] = useState({
        bloom: { enabled: false, threshold: 0.8, weight: 0.3, kernel: 64 },
        fxaa: { enabled: true },
        grain: { enabled: false, intensity: 10 },
        chromaticAberration: { enabled: false, aberrationAmount: 30 },
    });
    const [selectedPreset, setSelectedPreset] = useState<string>('');
    const [llmProvider, setLlmProvider] = useState<'gemini' | 'local' | 'lmstudio'>(() => getInitialState('shadercraft_llm_provider', 'gemini'));
    const [localLlmEndpoint, setLocalLlmEndpoint] = useState<string>(() => getInitialState('shadercraft_llm_endpoint', 'http://localhost:11434/api/generate'));
    const [localLlmModel, setLocalLlmModel] = useState<string>(() => getInitialState('shadercraft_llm_model', 'codellama'));
    const [localLlmStatus, setLocalLlmStatus] = useState<'unchecked' | 'connected' | 'error'>('unchecked');
    const [lmStudioUrl, setLmStudioUrl] = useState<string>(() => getInitialState('shadercraft_lmstudio_url', 'http://192.168.68.56:1234'));
    const [lmStudioStatus, setLmStudioStatus] = useState<'unchecked' | 'connected' | 'error'>('unchecked');
    const [selectedLmStudioModel, setSelectedLmStudioModel] = useState<string>(() => getInitialState('shadercraft_lmstudio_model', ''));
    const [lmStudioModels, setLmStudioModels] = useState<string[]>([]);
    const [isFetchingLmStudioModels, setIsFetchingLmStudioModels] = useState<boolean>(false);


    // State for AI Refinement
    const [hasSelection, setHasSelection] = useState<boolean>(false);
    const [isRefining, setIsRefining] = useState<boolean>(false);
    const [refineModalOpen, setRefineModalOpen] = useState<boolean>(false);
    const [refinementPrompt, setRefinementPrompt] = useState<string>('');
    const [refinementSelection, setRefinementSelection] = useState<RefinementSelection | null>(null);

    // State for Time Control
    const [timeState, setTimeState] = useState({ playing: true, time: 0.0 });
    
    // Unique error message identifiers for structured feedback
    const GEMINI_RATE_LIMIT_ERROR_MESSAGE = 'GEMINI_RATE_LIMIT_ERROR';
    const LMSTUDIO_CONNECTION_ERROR_MESSAGE = `LMSTUDIO_CONNECTION_ERROR`;
    const LOCAL_LLM_CONNECTION_ERROR_MESSAGE = "LOCAL_LLM_CONNECTION_ERROR";
    const TIMEOUT_ERROR_MESSAGE = "TIMEOUT_ERROR";


    const babylonCanvas = useRef<HTMLCanvasElement | null>(null);
    const sceneRef = useRef<any>(null);
    const engineRef = useRef<any>(null);
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

    // Keep refs in sync with the latest state to avoid stale closures
    useEffect(() => {
        lightStateRef.current = lightState;
    }, [lightState]);

    useEffect(() => {
        materialStateRef.current = materialState;
    }, [materialState]);

    useEffect(() => {
        timeStateRef.current = timeState;
    }, [timeState]);


    // Save LLM settings to localStorage
    useEffect(() => {
        localStorage.setItem('shadercraft_llm_provider', JSON.stringify(llmProvider));
        localStorage.setItem('shadercraft_llm_endpoint', JSON.stringify(localLlmEndpoint));
        localStorage.setItem('shadercraft_llm_model', JSON.stringify(localLlmModel));
        localStorage.setItem('shadercraft_lmstudio_url', JSON.stringify(lmStudioUrl));
        localStorage.setItem('shadercraft_lmstudio_model', JSON.stringify(selectedLmStudioModel));
    }, [llmProvider, localLlmEndpoint, localLlmModel, lmStudioUrl, selectedLmStudioModel]);

    // Test local LLM (Ollama) connection
    useEffect(() => {
        if (llmProvider !== 'local' || !localLlmEndpoint) {
            setLocalLlmStatus('unchecked');
            return;
        };

        const controller = new AbortController();
        const timeoutId = setTimeout(async () => {
            try {
                // Use a simple HEAD or OPTIONS request to check for server availability without sending a full prompt.
                // This is lighter and faster. Some servers might not support it, so a fallback is good.
                const response = await fetch(localLlmEndpoint, {
                    method: 'HEAD', // or 'OPTIONS'
                    signal: controller.signal,
                });
                
                // A successful response (even 405 Method Not Allowed) means the server is running.
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
        }, 500); // Debounce for 500ms

        return () => {
            clearTimeout(timeoutId);
            controller.abort();
        };
    }, [localLlmEndpoint, llmProvider]);

    const fetchLmStudioModels = useCallback(async () => {
        if (!lmStudioUrl || llmProvider !== 'lmstudio') {
            setLmStudioStatus('unchecked');
            setLmStudioModels([]);
            return;
        }
        
        setIsFetchingLmStudioModels(true);
        setLmStudioStatus('unchecked'); // Show as checking
        setError(''); // Clear previous errors

        try {
            const url = new URL('/v1/models', lmStudioUrl).toString();
            const response = await fetch(url, {
                signal: AbortSignal.timeout(15000) // 15 second timeout for fetching models
            });
            
            if (!response.ok) throw new Error(`Server responded with status: ${response.status}`);

            const data = await response.json();
            const models = data.data?.map((model: any) => model.id) || [];
            
            if (models.length === 0) throw new Error("No models found on the server.");

            setLmStudioModels(models);
            setLmStudioStatus('connected');
            
            // If current selection is not valid, select the first model
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
    }, [lmStudioUrl, llmProvider, LMSTUDIO_CONNECTION_ERROR_MESSAGE, TIMEOUT_ERROR_MESSAGE]);

    // Effect to auto-fetch LM Studio models when URL or provider changes
    useEffect(() => {
        if (llmProvider === 'lmstudio') {
            const timeoutId = setTimeout(() => {
                fetchLmStudioModels();
            }, 500); // Debounce
            return () => clearTimeout(timeoutId);
        } else {
             setLmStudioStatus('unchecked');
             setLmStudioModels([]);
        }
    }, [lmStudioUrl, llmProvider, fetchLmStudioModels]);

    // Load saved shaders from localStorage on initial mount
    useEffect(() => {
        setSavedShaders(getInitialState('shadercraft_shaders', []));
    }, []);
    
    // Clear error message when switching AI provider
    useEffect(() => {
        setError('');
    }, [llmProvider]);

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
            // Update existing shader
            updatedShaders = [...savedShaders];
            updatedShaders[existingShaderIndex] = newShader;
        } else {
            // Add new shader
            updatedShaders = [...savedShaders, newShader];
        }

        setSavedShaders(updatedShaders);
        localStorage.setItem('shadercraft_shaders', JSON.stringify(updatedShaders));
        setShaderName('');
        // Ensure the newly saved/updated shader is selected in the dropdown
        setSelectedShader(newShader.name);
        alert(`Shader "${newShader.name}" saved!`);
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
                // Reset to default for older saved shaders without material properties
                setMaterialState({ albedo: '#b3b3b3', metallic: 0.1, roughness: 0.5 });
            }
            setSelectedPreset(''); // Clear preset selection when loading a saved shader
        }
    };

    const handleDeleteShader = () => {
        if (!selectedShader) {
            alert("Please select a shader to delete.");
            return;
        }
        if (window.confirm(`Are you sure you want to delete the shader "${selectedShader}"?`)) {
            const updatedShaders = savedShaders.filter(s => s.name !== selectedShader);
            setSavedShaders(updatedShaders);
            localStorage.setItem('shadercraft_shaders', JSON.stringify(updatedShaders));
            setSelectedShader(''); // Reset selection
        }
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

        const jsonString = JSON.stringify(packageData, null, 2); // Pretty print
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

                // Validate the structure of the imported JSON
                if (!importedData.name || typeof importedData.name !== 'string' ||
                    !importedData.vertexShader || typeof importedData.vertexShader !== 'string' ||
                    !importedData.fragmentShader || typeof importedData.fragmentShader !== 'string') {
                    throw new Error("Invalid shader format. JSON must contain 'name', 'vertexShader', and 'fragmentShader' properties.");
                }

                const newShader: SavedShader = {
                    name: importedData.name.trim(),
                    vertex: importedData.vertexShader,
                    fragment: importedData.fragmentShader,
                    material: importedData.material // Will be undefined if not present, which is fine
                };

                const existingShaderIndex = savedShaders.findIndex(s => s.name === newShader.name);
                let updatedShaders;

                if (existingShaderIndex > -1) {
                    // If a shader with the same name exists, ask for confirmation to overwrite
                    if (!window.confirm(`A shader named "${newShader.name}" already exists. Do you want to overwrite it?`)) {
                        return; // User canceled the overwrite
                    }
                    updatedShaders = [...savedShaders];
                    updatedShaders[existingShaderIndex] = newShader;
                } else {
                    // Add the new shader to the list
                    updatedShaders = [...savedShaders, newShader];
                }

                setSavedShaders(updatedShaders);
                localStorage.setItem('shadercraft_shaders', JSON.stringify(updatedShaders));
                
                // Automatically select and load the newly imported shader for immediate use
                setSelectedShader(newShader.name);
                setVertexCode(newShader.vertex);
                setFragmentCode(newShader.fragment);
                if (newShader.material) {
                    setMaterialState(newShader.material);
                }
                
                alert(`Shader "${newShader.name}" imported successfully!`);

            } catch (err: any) {
                console.error("Failed to import shader:", err);
                setError(err instanceof Error ? err.message : "An unknown error occurred during import.");
            } finally {
                // Reset file input to allow re-uploading the same file if needed
                event.target.value = '';
            }
        };

        reader.onerror = () => {
            setError("Failed to read the selected file.");
        };

        reader.readAsText(file);
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

        shaderMaterial.onCompiled = () => {
             console.log("Shader compiled successfully");
        };

    }, [vertexCode, fragmentCode]);

    // Effect for time animation using requestAnimationFrame
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
    }, []); // This effect should only run once on mount

    // Effect for one-time Babylon scene setup
    useEffect(() => {
        if (!babylonCanvas.current) return;

        const engine = new BABYLON.Engine(babylonCanvas.current, true);
        engineRef.current = engine;
        const scene = new BABYLON.Scene(engine);
        sceneRef.current = scene;

        const camera = new BABYLON.ArcRotateCamera("camera", -Math.PI / 2, Math.PI / 2.5, 5, BABYLON.Vector3.Zero(), scene);
        camera.attachControl(babylonCanvas.current, true);
        
        // Setup Post-Processing Pipeline
        const defaultPipeline = new BABYLON.DefaultRenderingPipeline(
            "defaultPipeline",
            true, // is HDR
            scene,
            [camera]
        );
        ppPipelineRef.current = defaultPipeline;

        engine.runRenderLoop(() => {
            const material = scene.getMaterialByName("customShader");
            if (material && material.getClassName() === "ShaderMaterial") {
                const ls = lightStateRef.current;
                const ms = materialStateRef.current;
                const lightVector = new BABYLON.Vector3(ls.direction.x, ls.direction.y, ls.direction.z);
                
                // Time and Material Uniforms
                (material as any).setFloat("u_time", timeStateRef.current.time);
                material.setColor3("u_albedo", BABYLON.Color3.FromHexString(ms.albedo));
                material.setFloat("u_metallic", ms.metallic);
                material.setFloat("u_roughness", ms.roughness);

                // Light Uniforms
                material.setFloat("u_lightIntensity", ls.intensity);
                material.setColor3("u_lightColor", BABYLON.Color3.FromHexString(ls.diffuse));

                if (ls.type === 'point') {
                    material.setInt("u_lightType", 1);
                    material.setVector3("u_lightPosition", lightVector);
                } else { // Hemispheric and Directional
                    material.setInt("u_lightType", 0);
                    material.setVector3("u_lightDirection", lightVector);
                }

                // Camera and Environment Uniforms
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
    }, []); // This effect should only run once on mount
    
    // Effect to create/update the mesh and apply the current shader
    useEffect(() => {
        if (!sceneRef.current) return;
        const scene = sceneRef.current;

        // Only recreate the mesh if the type has changed or it doesn't exist yet.
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
                    // Use Ground to allow for subdivisions
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
        
        // Always run the shader logic to apply the latest code to the current mesh.
        handleRunShader();

        // Update the ref to track the current mesh type for the next run.
        prevSelectedMeshRef.current = selectedMesh;

    }, [selectedMesh, meshResolution, handleRunShader]);

    // Effect to toggle wireframe on the mesh material
    useEffect(() => {
        if (meshRef.current && meshRef.current.material) {
            meshRef.current.material.wireframe = showWireframe;
        }
    }, [showWireframe, meshRef.current?.material]); // Re-run if wireframe is toggled or material changes

    // Effect for live reload functionality
    useEffect(() => {
        if (!liveReload) return;
    
        const handler = setTimeout(() => {
            handleRunShader();
        }, 500);
    
        return () => {
            clearTimeout(handler);
        };
    }, [vertexCode, fragmentCode, liveReload, handleRunShader]);

    // Effect to manage the scene's light
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

        // If light type changes or light doesn't exist, (re)create it
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

        // Update light properties
        const light = lightRef.current;
        if (light) {
            // NOTE: Babylon's light intensity isn't used by our shader, we pass it directly.
            light.diffuse = BABYLON.Color3.FromHexString(lightState.diffuse);
            const vector = new BABYLON.Vector3(lightState.direction.x, lightState.direction.y, lightState.direction.z);
            
            if (light.direction) { // For Hemispheric, Directional
                light.direction = vector;
            }
            if (light.position) { // For Point
                light.position = vector;
            }
        }

    }, [lightState]);

    // Effect to manage environment texture and skybox
    useEffect(() => {
        if (!sceneRef.current) return;
        const scene = sceneRef.current;

        // Local variables to hold the resources created in this effect run
        let createdSkybox: any = null;
        let createdTexture: any = null;
        let textureUrlToRevoke: string | null = null;

        // Set up new texture and skybox if an environmentTexture is provided
        if (environmentTexture) {
            createdTexture = new BABYLON.EquiRectangularCubeTexture(environmentTexture, scene, 512);
            scene.environmentTexture = createdTexture;
            createdSkybox = scene.createDefaultSkybox(createdTexture, true, 1000, 0.5);
            skyboxRef.current = createdSkybox;
            
            if (environmentTexture.startsWith('blob:')) {
                textureUrlToRevoke = environmentTexture;
            }
        } else {
            // If no texture is provided, ensure the scene's texture and our ref are null
            scene.environmentTexture = null;
            skyboxRef.current = null;
        }

        // The cleanup function will run when the dependency changes, or on unmount.
        // It's responsible for disposing of the resources created in *this specific* effect run.
        return () => {
            if (createdSkybox) {
                createdSkybox.dispose();
            }
            if (createdTexture) {
                createdTexture.dispose();
            }
            if (textureUrlToRevoke) {
                URL.revokeObjectURL(textureUrlToRevoke);
            }
        };
    }, [environmentTexture]);

    // Effect to control post-processing based on state
    useEffect(() => {
        const pipeline = ppPipelineRef.current;
        if (!pipeline) return;

        // Bloom
        pipeline.bloomEnabled = postProcessingState.bloom.enabled;
        if (pipeline.bloomEnabled) {
            pipeline.bloomThreshold = postProcessingState.bloom.threshold;
            pipeline.bloomWeight = postProcessingState.bloom.weight;
            pipeline.bloomKernel = postProcessingState.bloom.kernel;
        }

        // FXAA
        pipeline.fxaaEnabled = postProcessingState.fxaa.enabled;

        // Grain
        pipeline.grainEnabled = postProcessingState.grain.enabled;
        if (pipeline.grainEnabled) {
            pipeline.grain.intensity = postProcessingState.grain.intensity;
            pipeline.grain.animated = true; // Keep it animated
        }

        // Chromatic Aberration
        pipeline.chromaticAberrationEnabled = postProcessingState.chromaticAberration.enabled;
        if (pipeline.chromaticAberrationEnabled) {
            pipeline.chromaticAberration.aberrationAmount = postProcessingState.chromaticAberration.aberrationAmount;
            pipeline.chromaticAberration.radialIntensity = 1; // Default
        }

    }, [postProcessingState]);


    // Effect for initializing CodeMirror editors and adding selection listener
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
                    // Manually trigger linting on change
                    instance.performLint();
                });
                cm.on('cursorActivity', (instance: any) => {
                    // This listener might fire for both editors, so we ensure hasSelection is true
                    // if *either* has a selection. A more robust solution might track them separately.
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

    // Sync state changes to CodeMirror editors
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

    // Refresh CodeMirror instance when its tab becomes visible
    useEffect(() => {
        setTimeout(() => {
            if (activeTab === 'vertex') vertexCmRef.current?.refresh();
            if (activeTab === 'fragment') fragmentCmRef.current?.refresh();
        }, 1);
    }, [activeTab]);

    const formatGlslCode = async (code: string): Promise<string> => {
        try {
            // Prettier and its plugins are loaded from the CDN and available globally
            return await prettier.format(code, {
                parser: 'glsl-parse',
                plugins: [prettierPlugins.glsl],
            });
        } catch (error) {
            console.warn('Prettier GLSL formatting failed:', error);
            return code; // Fallback to unformatted code on error
        }
    };

    const handleFormatCode = async () => {
        if (activeTab === 'vertex') {
            const formatted = await formatGlslCode(vertexCode);
            setVertexCode(formatted);
        } else {
            const formatted = await formatGlslCode(fragmentCode);
            setFragmentCode(formatted);
        }
    };
    
    // Extracts a JSON block from a string that might be wrapped in markdown or have extraneous text.
    const extractJsonFromString = (str: string): string | null => {
        // First, try to find a JSON block within markdown fences
        const markdownMatch = str.match(/```json\s*([\s\S]*?)\s*```/);
        if (markdownMatch && markdownMatch[1]) {
            return markdownMatch[1].trim();
        }

        // If not found, look for the substring between the first '{' and the last '}'
        // This is a robust way to handle extraneous text before or after the JSON object.
        const firstBraceIndex = str.indexOf('{');
        const lastBraceIndex = str.lastIndexOf('}');

        if (firstBraceIndex !== -1 && lastBraceIndex > firstBraceIndex) {
            return str.substring(firstBraceIndex, lastBraceIndex + 1).trim();
        }
        
        // Return null if no JSON object could be extracted.
        return null;
    };

    const handleAiError = (error: any) => {
        console.error("AI Error:", error);
        const errorMessage = error instanceof Error ? error.message : String(error);

        if (error.name === 'TimeoutError' || (error instanceof DOMException && error.name === 'AbortError')) {
            setError(TIMEOUT_ERROR_MESSAGE);
            return;
        }
    
        if (llmProvider === 'gemini' && (errorMessage.includes('429') || errorMessage.includes('RESOURCE_EXHAUSTED'))) {
            setError(GEMINI_RATE_LIMIT_ERROR_MESSAGE);
            return;
        }
    
        if (error instanceof TypeError && errorMessage === 'Failed to fetch') {
            if (llmProvider === 'lmstudio') {
                setError(LMSTUDIO_CONNECTION_ERROR_MESSAGE);
            } else if (llmProvider === 'local') {
                setError(LOCAL_LLM_CONNECTION_ERROR_MESSAGE);
            } else {
                 setError(`Connection failed: ${errorMessage}`);
            }
            return;
        }
        
        setError(`An error occurred: ${errorMessage}`);
    };

    const handleGenerateShader = async () => {
        setIsLoading(true);
        setError('');

        const isRefinement = vertexCode !== DEFAULT_VERTEX_SHADER || fragmentCode !== DEFAULT_FRAGMENT_SHADER;

        // FIX: Separated system instruction from user content for better prompting.
        const systemInstruction = `You are an expert in GLSL and Babylon.js. Create GLSL shaders that will run within a Babylon.js ShaderMaterial.

Please provide the complete GLSL code for both the vertex and fragment shaders.

- CRITICAL: DO NOT include the \`#version\` directive (e.g., \`#version 300 es\`) at the top of the shader code. Babylon.js handles this automatically.
- The vertex shader MUST define \`gl_Position\`.
- It will receive attributes: \`vec3 position\`, \`vec3 normal\`, \`vec2 uv\`.
- It MUST pass a varying \`vUV\` (\`vec2\`), \`vNormal\` (\`vec3\`), and \`vPositionW\` (\`vec3\`) to the fragment shader.
- The fragment shader MUST define \`gl_FragColor\`.
- It will receive the varyings \`vUV\`, \`vNormal\`, and \`vPositionW\`.
- Babylon.js provides these uniforms automatically: \`mat4 worldViewProjection\`, \`mat4 world\`, \`mat4 view\`, \`mat4 projection\`.
- Custom uniforms are also provided: \`float u_time\`, \`vec3 u_lightColor\`, \`float u_lightIntensity\`, \`vec3 u_lightDirection\`, \`vec3 u_lightPosition\`, \`int u_lightType\`, \`vec3 u_cameraPosition\`.
- PBR uniforms are: \`vec3 u_albedo\`, \`float u_metallic\`, \`float u_roughness\`. You can use these instead of hard-coding material properties.
- Environment reflection uniforms are: \`samplerCube u_envTexture\`, \`int u_hasEnvTexture\`. The shader MUST use \`u_hasEnvTexture\` to conditionally apply reflections.
- The shader MUST use \`u_lightType\` to differentiate between directional/hemispheric (0) and point (1) lights.
- CRITICAL: The returned GLSL code must be thoroughly commented to explain complex logic, uniform variables, and the overall purpose of different code blocks.

Return ONLY the code in a JSON object with keys "vertexShader" and "fragmentShader". Do not include any extra explanations or markdown formatting. The JSON object must be valid.`;

        const userContent = `The user wants a shader with this effect: "${prompt}"
${
  isRefinement
    ? `The user wants to refine the following existing shaders. Modify them to achieve the desired effect.
Current Vertex Shader:
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
            const timeoutId = setTimeout(() => controller.abort(), 60000); // 60 second timeout for generation

            if (llmProvider === 'gemini') {
                const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
                const response = await ai.models.generateContent({
                    model: "gemini-2.5-flash",
                    contents: userContent,
                    config: {
                        systemInstruction: systemInstruction,
                        responseMimeType: "application/json",
                        responseSchema: SHADER_SCHEMA,
                    }
                });
                rawResponseText = response.text.trim();

            } else if (llmProvider === 'lmstudio') {
                if (lmStudioStatus !== 'connected' || !selectedLmStudioModel) {
                    throw new Error(LMSTUDIO_CONNECTION_ERROR_MESSAGE);
                }
                const url = new URL('/v1/chat/completions', lmStudioUrl).toString();
                const response = await fetch(url, {
                    method: 'POST',
                    signal: controller.signal,
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        model: selectedLmStudioModel,
                        messages: [
                            { role: 'system', content: systemInstruction },
                            { role: 'user', content: userContent }
                        ],
                        stream: false,
                        // response_format: { type: 'json_object' } // Removed for better compatibility
                    })
                });
                if (!response.ok) {
                    const errorText = await response.text();
                    throw new Error(`LM Studio request failed: ${response.statusText} - ${errorText}`);
                }
                const responseData = await response.json();
                const content = responseData.choices?.[0]?.message?.content;
                if (!content) {
                    throw new Error("LM Studio returned an empty or invalid response structure.");
                }
                rawResponseText = content;
            } else { // Local LLM (Ollama) provider
                if (localLlmStatus !== 'connected') {
                    throw new Error(LOCAL_LLM_CONNECTION_ERROR_MESSAGE);
                }
                const response = await fetch(localLlmEndpoint, {
                    method: 'POST',
                    signal: controller.signal,
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        model: localLlmModel,
                        prompt: `${systemInstruction}\n\n${userContent}`, // Combine for local models
                        stream: false,
                        format: 'json' // Some servers like Ollama support this
                    })
                });
                if (!response.ok) {
                    throw new Error(`Local LLM request failed: ${response.statusText}`);
                }
                const responseData = await response.json();
                
                // Response structure can vary (e.g., { response: "..." } for Ollama)
                rawResponseText = responseData.response || responseData.content || JSON.stringify(responseData);
            }

            clearTimeout(timeoutId);

            if (!rawResponseText) {
                throw new Error("AI response was empty or malformed.");
            }

            // Centralized JSON extraction and parsing to handle responses wrapped in markdown
            const jsonString = extractJsonFromString(rawResponseText) || rawResponseText;
            const shaderData = JSON.parse(jsonString);

            if (shaderData.vertexShader && shaderData.fragmentShader) {
                const formattedVertex = await formatGlslCode(shaderData.vertexShader);
                const formattedFragment = await formatGlslCode(shaderData.fragmentShader);
                setVertexCode(formattedVertex);
                setFragmentCode(formattedFragment);
                setSelectedPreset(''); // Clear preset selection after generating
                 // Reset material to a neutral default when generating a new shader
                setMaterialState({ albedo: '#b3b3b3', metallic: 0.1, roughness: 0.5 });
            } else {
                setError("AI response was missing shader code. Please try again.");
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
            ? "Generate a single, short, and creative prompt for a GLSL shader effect. Be descriptive and inspiring. Draw from themes like: glows, flows, organic growth, waves, tiles, natural textures (wood, stone), metals, and intricate patterns. Examples: 'shimmering iridescent fish scales', 'molten lava slowly cracking', 'glowing alien circuits', 'polished mahogany wood grain', 'rippling water with caustic patterns', 'swirling magical energy shield'. Return ONLY the raw text content for the prompt. Do not include any extra words, formatting, escaped symbols, or XML data."
            : `You are a creative assistant for a 3D artist. Take the following shader idea and enhance it, making it more descriptive, vivid, and inspiring, but keep it as a concise prompt. User's idea: "${prompt}". Return ONLY the raw, enhanced prompt text. Do not include any introductory phrases, escaped symbols, or XML data.`;
        
        try {
            let resultText: string;
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 20000); // 20 second timeout for prompt actions

            if (llmProvider === 'gemini') {
                const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
                const response = await ai.models.generateContent({
                    model: "gemini-2.5-flash",
                    contents: content,
                    // Add temperature for more creative random prompts
                    config: {
                        temperature: action === 'random' ? 0.9 : undefined,
                    }
                });
                resultText = response.text;
            } else if (llmProvider === 'lmstudio') {
                if (lmStudioStatus !== 'connected' || !selectedLmStudioModel) {
                    throw new Error(LMSTUDIO_CONNECTION_ERROR_MESSAGE);
                }
                const url = new URL('/v1/chat/completions', lmStudioUrl).toString();
                const requestBody: any = {
                    model: selectedLmStudioModel,
                    messages: [{ role: 'user', content: content }],
                    stream: false
                };
                if (action === 'random') {
                    requestBody.temperature = 0.9;
                }
                const response = await fetch(url, {
                    method: 'POST',
                    signal: controller.signal,
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(requestBody)
                });
                if (!response.ok) {
                    const errorText = await response.text();
                    throw new Error(`LM Studio request failed: ${response.statusText} - ${errorText}`);
                }
                const data = await response.json();
                resultText = data.choices?.[0]?.message?.content || '';
            } else { // Local LLM (Ollama)
                if (localLlmStatus !== 'connected') {
                    throw new Error(LOCAL_LLM_CONNECTION_ERROR_MESSAGE);
                }

                const requestBody: any = {
                    model: localLlmModel,
                    prompt: content,
                    stream: false
                };
                if (action === 'random') {
                    // Add temperature for more creative random prompts
                    // Note: The property might be 'temperature' or inside 'options' depending on the local server.
                    requestBody.temperature = 0.9;
                }

                const response = await fetch(localLlmEndpoint, {
                     method: 'POST',
                     signal: controller.signal,
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(requestBody)
                });
                if (!response.ok) throw new Error(`Local LLM request failed: ${response.statusText}`);
                const data = await response.json();
                resultText = data.response || data.content || '';
            }
            
            clearTimeout(timeoutId);
            setPrompt(resultText.trim().replace(/['"]+/g, '')); // Clean up quotes
        } catch (e: any) {
            handleAiError(e);
        } finally {
            setPromptActionLoading(null);
        }
    };

    const toggleInspector = () => {
        if (sceneRef.current) {
            if (sceneRef.current.debugLayer.isVisible()) {
                sceneRef.current.debugLayer.hide();
            } else {
                sceneRef.current.debugLayer.show({ embedMode: true });
            }
        }
    };

    const handleOpenRefineModal = () => {
        const cm = activeTab === 'vertex' ? vertexCmRef.current : fragmentCmRef.current;
        if (cm && cm.somethingSelected()) {
            setRefinementSelection({
                code: cm.getSelection(),
                editor: activeTab
            });
            setRefinementPrompt(''); // Clear previous prompt
            setRefineModalOpen(true);
        }
    };

    const closeRefineModal = () => {
        setRefineModalOpen(false);
        setRefinementSelection(null);
    };

    const handleRefineCode = async () => {
        if (!refinementSelection || !refinementPrompt) return;
        setIsRefining(true);
        setError('');

        const fullShaderCode = refinementSelection.editor === 'vertex' ? vertexCode : fragmentCode;

        const systemInstruction = `You are an expert GLSL code assistant. Your task is to rewrite a selected piece of GLSL code based on a user's instruction.
IMPORTANT: You must return ONLY the raw, modified GLSL code snippet. Do not wrap it in markdown, do not add any comments that were not in the original selection unless requested, and do not add any explanatory text before or after the code.`;
        
        const userContent = `The user wants to modify a piece of code.
Instruction: "${refinementPrompt}"

This is the full ${refinementSelection.editor} shader for context:
\`\`\`glsl
${fullShaderCode}
\`\`\`

This is the specific snippet to modify:
\`\`\`glsl
${refinementSelection.code}
\`\`\`
`;
        try {
            let refinedCode: string;
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 45000); // 45 second timeout for refinement

            if (llmProvider === 'gemini') {
                const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
                const response = await ai.models.generateContent({
                    model: "gemini-2.5-flash",
                    contents: userContent,
                    config: {
                        systemInstruction: systemInstruction,
                    },
                });
                refinedCode = response.text.trim();
            } else if (llmProvider === 'lmstudio') {
                if (lmStudioStatus !== 'connected' || !selectedLmStudioModel) {
                     throw new Error(LMSTUDIO_CONNECTION_ERROR_MESSAGE);
                }
                const url = new URL('/v1/chat/completions', lmStudioUrl).toString();
                const response = await fetch(url, {
                    method: 'POST',
                    signal: controller.signal,
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        model: selectedLmStudioModel,
                        messages: [
                            { role: 'system', content: systemInstruction },
                            { role: 'user', content: userContent }
                        ],
                        stream: false,
                    })
                });
                if (!response.ok) {
                    const errorText = await response.text();
                    throw new Error(`LM Studio request failed: ${response.statusText} - ${errorText}`);
                }
                const data = await response.json();
                const content = data.choices?.[0]?.message?.content || '';

                // Defensively strip markdown in case the model adds it
                const markdownMatch = content.match(/```(?:glsl)?\s*([\s\S]*?)\s*```/);
                refinedCode = markdownMatch ? markdownMatch[1].trim() : content.trim();

            } else { // Local LLM (Ollama)
                if (localLlmStatus !== 'connected') {
                    throw new Error(LOCAL_LLM_CONNECTION_ERROR_MESSAGE);
                }
                 const response = await fetch(localLlmEndpoint, {
                     method: 'POST',
                     signal: controller.signal,
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ model: localLlmModel, prompt: `${systemInstruction}\n\n${userContent}`, stream: false })
                });
                if (!response.ok) throw new Error(`Local LLM request failed: ${response.statusText}`);
                const data = await response.json();
                const rawResponse = data.response || data.content || '';
                
                // Also strip markdown for local models for consistency
                const markdownMatch = rawResponse.match(/```(?:glsl)?\s*([\s\S]*?)\s*```/);
                refinedCode = markdownMatch ? markdownMatch[1].trim() : rawResponse.trim();
            }
            
            clearTimeout(timeoutId);

            if (!refinedCode) {
                 throw new Error("AI returned an empty response.");
            }
            
            // Replace the selection in the correct editor
            const cm = refinementSelection.editor === 'vertex' ? vertexCmRef.current : fragmentCmRef.current;
            if (cm) {
                cm.replaceSelection(refinedCode);
            }

            closeRefineModal();

        } catch (e: any) {
            handleAiError(e);
            // Don't close the modal on error, so the user can try again
        } finally {
            setIsRefining(false);
        }
    };
    
    // --- Drag and Drop Handlers for Control Panels ---
    const handleDragStart = (e: React.DragEvent<HTMLSpanElement>, position: number) => {
        dragItem.current = position;
        // Add a class to the panel being dragged for visual feedback
        setTimeout(() => {
            const panel = (e.target as HTMLElement).closest('.collapsible-panel');
            panel?.classList.add('dragging');
        }, 0);
    };

    const handleDragEnter = (e: React.DragEvent<HTMLDivElement>, position: number) => {
        dragOverItem.current = position;
    };

    const handleDrop = () => {
        if (dragItem.current === null || dragOverItem.current === null || dragItem.current === dragOverItem.current) {
            return; // No change
        }
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
    
    // --- Collapse/Expand Handler ---
    const togglePanel = (key: string) => {
        setCollapsedPanels(prev => ({ ...prev, [key]: !prev[key] }));
    };

    // --- Environment Texture Handlers ---
    const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (e) => {
                setEnvironmentTexture(e.target?.result as string);
            };
            reader.readAsDataURL(file);
        }
        // Reset file input to allow re-uploading the same file
        event.target.value = '';
    };

    const handleRandomBackground = async () => {
        setError('');
        try {
            // Using picsum.photos as it's more reliable for this kind of hotlinking.
            const randomImageUrl = `https://picsum.photos/2048/1024?random=${Date.now()}`;
            const response = await fetch(randomImageUrl);
            if (!response.ok) {
                throw new Error(`Failed to fetch image: ${response.statusText}`);
            }
            const imageBlob = await response.blob();
            // Create a local URL for the blob to bypass CORS issues
            const objectUrl = URL.createObjectURL(imageBlob);
            setEnvironmentTexture(objectUrl);
        } catch (err: any) {
            console.error("Error fetching random background:", err);
            setError(err instanceof Error ? err.message : "Could not load random background image.");
        }
    };

    const clearEnvironment = () => {
        setEnvironmentTexture(null);
    };

    // --- Preset Handler ---
    const handlePresetChange = (presetName: string) => {
        setSelectedPreset(presetName);
        if (!presetName) return;

        const preset = SHADER_PRESETS.find(p => p.name === presetName);
        if (preset) {
            setMaterialState({
                albedo: preset.albedo,
                metallic: preset.metallic,
                roughness: preset.roughness,
            });
            setSelectedShader(''); // Clear saved shader selection
        }
    };

    // --- Time Control Handlers ---
    const handleTogglePlay = () => {
        setTimeState(prev => ({ ...prev, playing: !prev.playing }));
    };

    const handleResetTime = () => {
        setTimeState(prev => ({ ...prev, time: 0.0, playing: prev.playing }));
    };

    const handleTimeScrub = (e: React.ChangeEvent<HTMLInputElement>) => {
        const newTime = parseFloat(e.target.value);
        setTimeState(prev => ({ ...prev, time: newTime }));
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
                            case 'ai':
                                title = 'AI Controls';
                                content = (
                                    <>
                                        <div className="form-group">
                                            <label htmlFor="llm-provider-select">AI Provider</label>
                                            <select
                                                id="llm-provider-select"
                                                value={llmProvider}
                                                onChange={(e) => setLlmProvider(e.target.value as 'gemini' | 'local' | 'lmstudio')}
                                            >
                                                <option value="gemini">Gemini API</option>
                                                <option value="lmstudio">LM Studio</option>
                                                <option value="local">Local LLM (Ollama)</option>
                                            </select>
                                        </div>
                                        
                                        {/* --- UNIFIED ERROR DISPLAY --- */}
                                        {error && (
                                            <>
                                                {error === GEMINI_RATE_LIMIT_ERROR_MESSAGE && (
                                                    <div className="error-message-inline structured-error">
                                                        <strong>Gemini API Rate Limit Exceeded</strong>
                                                        <p>You've made too many requests in a short period. Please check your plan and billing details, or wait a minute before trying again.</p>
                                                        <p>For more information, see the <a href="https://ai.google.dev/gemini-api/docs/rate-limits" target="_blank" rel="noopener noreferrer">Gemini API Rate Limits documentation</a>.</p>
                                                    </div>
                                                )}
                                                {error === LMSTUDIO_CONNECTION_ERROR_MESSAGE && (
                                                    <div className="error-message-inline structured-error">
                                                        <strong>LM Studio Connection Failed</strong>
                                                        <p>This is a network issue, most likely related to CORS. Please check the following:</p>
                                                        <ul>
                                                            <li>In the LM Studio app, go to the "Server" tab and <strong>check the "Enable CORS" box</strong>. This is the most common fix.</li>
                                                            <li>Ensure the LM Studio server is running.</li>
                                                            <li>Verify the Server URL below is correct.</li>
                                                            <li>Check for firewalls blocking the connection.</li>
                                                        </ul>
                                                    </div>
                                                )}
                                                {error === LOCAL_LLM_CONNECTION_ERROR_MESSAGE && (
                                                    <div className="error-message-inline structured-error">
                                                        <strong>Local LLM Connection Failed</strong>
                                                        <p>This is a network issue. Please check the following:</p>
                                                        <ul>
                                                            <li>Is your local server (e.g., Ollama) running?</li>
                                                            <li>Is the Endpoint URL below correct?</li>
                                                            <li>Your server may need to be configured to allow requests from this web page (CORS).</li>
                                                        </ul>
                                                    </div>
                                                )}
                                                {error === TIMEOUT_ERROR_MESSAGE && (
                                                    <div className="error-message-inline structured-error">
                                                        <strong>Request Timed Out</strong>
                                                        <p>The request to the AI model took too long to respond. This can happen for several reasons:</p>
                                                        <ul>
                                                            <li>The AI model is very large and is still loading into memory.</li>
                                                            <li>The task is very complex and requires a lot of computation.</li>
                                                            <li>Your network connection to the server is slow.</li>
                                                        </ul>
                                                         <p>Please try again in a moment. If the problem persists, consider using a smaller model.</p>
                                                    </div>
                                                )}
                                                {/* Fallback for other, unexpected errors */}
                                                {![GEMINI_RATE_LIMIT_ERROR_MESSAGE, LMSTUDIO_CONNECTION_ERROR_MESSAGE, LOCAL_LLM_CONNECTION_ERROR_MESSAGE, TIMEOUT_ERROR_MESSAGE].includes(error) && (
                                                    <pre className="error-message-inline">{error}</pre>
                                                )}
                                            </>
                                        )}

                                        <div className={`effect-options ${llmProvider === 'lmstudio' ? 'visible' : ''}`}>
                                            <div>
                                                <div className="form-group">
                                                    <label htmlFor="lmstudio-url">Server URL</label>
                                                    <div className="input-with-status">
                                                        <input
                                                            id="lmstudio-url"
                                                            type="text"
                                                            value={lmStudioUrl}
                                                            onChange={(e) => setLmStudioUrl(e.target.value)}
                                                            placeholder="http://192.168.68.56:1234"
                                                        />
                                                        <span className={`connection-status ${lmStudioStatus}`} title={
                                                            lmStudioStatus === 'connected' ? 'Connected' :
                                                            lmStudioStatus === 'error' ? 'Connection Failed' : 'Checking...'
                                                        }></span>
                                                    </div>
                                                </div>
                                                <div className="form-group">
                                                    <div className="prompt-label-group">
                                                        <label htmlFor="lmstudio-model">Model</label>
                                                        <button onClick={() => fetchLmStudioModels()} disabled={isFetchingLmStudioModels || !lmStudioUrl} className="button-small">
                                                            {isFetchingLmStudioModels ? <span className="loader" /> : 'Refresh'}
                                                        </button>
                                                    </div>
                                                    <select
                                                        id="lmstudio-model"
                                                        value={selectedLmStudioModel}
                                                        onChange={(e) => setSelectedLmStudioModel(e.target.value)}
                                                        disabled={isFetchingLmStudioModels || lmStudioModels.length === 0}
                                                    >
                                                        {isFetchingLmStudioModels && <option>Fetching models...</option>}
                                                        {!isFetchingLmStudioModels && lmStudioStatus === 'error' && <option>Could not load models</option>}
                                                        {!isFetchingLmStudioModels && lmStudioStatus === 'connected' && lmStudioModels.length === 0 && <option>No models found</option>}
                                                        {lmStudioModels.map(model => (
                                                            <option key={model} value={model}>{model}</option>
                                                        ))}
                                                    </select>
                                                </div>
                                                <p className="form-hint">
                                                    Connects to your LM Studio server. <strong>Note:</strong> You must check the "Enable CORS" box in the LM Studio server settings.
                                                </p>
                                            </div>
                                        </div>
                                        
                                        <div className={`effect-options ${llmProvider === 'local' ? 'visible' : ''}`}>
                                            <div>
                                                <div className="form-group">
                                                    <label htmlFor="local-llm-endpoint">Endpoint URL</label>
                                                    <div className="input-with-status">
                                                        <input
                                                            id="local-llm-endpoint"
                                                            type="text"
                                                            value={localLlmEndpoint}
                                                            onChange={(e) => setLocalLlmEndpoint(e.target.value)}
                                                            placeholder="http://localhost:11434/api/generate"
                                                        />
                                                        <span className={`connection-status ${localLlmStatus}`} title={
                                                            localLlmStatus === 'connected' ? 'Connected' : 
                                                            localLlmStatus === 'error' ? 'Connection Failed' : 'Unchecked'
                                                        }></span>
                                                    </div>
                                                </div>
                                                <div className="form-group">
                                                    <label htmlFor="local-llm-model">Model Name</label>
                                                    <input
                                                        id="local-llm-model"
                                                        type="text"
                                                        value={localLlmModel}
                                                        onChange={(e) => setLocalLlmModel(e.target.value)}
                                                        placeholder="e.g., codellama"
                                                    />
                                                </div>
                                                <p className="form-hint">
                                                    Connects to your local LLM server (e.g., Ollama).
                                                </p>
                                            </div>
                                        </div>

                                        <div className="form-group">
                                            <div className="prompt-label-group">
                                                <label htmlFor="prompt-input">Shader Prompt</label>
                                                <div className="prompt-actions">
                                                    <button onClick={() => handlePromptAction('random')} disabled={!!promptActionLoading || isLoading} className="button-small" aria-label="Generate random prompt">
                                                        {promptActionLoading === 'random' ? <span className="loader" /> : 'Random'}
                                                    </button>
                                                    <button onClick={() => handlePromptAction('enhance')} disabled={!prompt || !!promptActionLoading || isLoading} className="button-small" aria-label="Enhance current prompt">
                                                         {promptActionLoading === 'enhance' ? <span className="loader" /> : 'Enhance'}
                                                    </button>
                                                </div>
                                            </div>
                                            <textarea
                                                id="prompt-input"
                                                value={prompt}
                                                onChange={(e) => setPrompt(e.target.value)}
                                                placeholder="e.g., a shiny metallic gold material, or a psychedelic rainbow effect..."
                                                aria-label="Enter your shader description here"
                                            />
                                        </div>
                                        <button onClick={handleGenerateShader} disabled={isLoading || !prompt}>
                                            {isLoading ? <span className="loader" /> : 'Generate with AI'}
                                        </button>
                                        <div className="form-group toggle-group">
                                            <label htmlFor="live-reload-toggle">Live Reload</label>
                                            <label className="switch">
                                                <input
                                                    id="live-reload-toggle"
                                                    type="checkbox"
                                                    checked={liveReload}
                                                    onChange={(e) => setLiveReload(e.target.checked)}
                                                    aria-checked={liveReload}
                                                />
                                                <span className="slider round"></span>
                                            </label>
                                        </div>
                                    </>
                                );
                                break;
                            case 'scene':
                                title = 'Scene Controls';
                                content = (
                                    <>
                                        <div className="form-group">
                                            <label htmlFor="mesh-select">Mesh Type</label>
                                            <select id="mesh-select" value={selectedMesh} onChange={(e) => setSelectedMesh(e.target.value)} aria-label="Select 3D object shape">
                                                <option value="sphere">Sphere</option>
                                                <option value="cube">Cube</option>
                                                <option value="torus">Torus</option>
                                                <option value="plane">Plane</option>
                                                <option value="cylinder">Cylinder</option>
                                            </select>
                                        </div>
                                        <div className="form-group form-group-row">
                                            <label htmlFor="mesh-resolution">Resolution</label>
                                            <input
                                                id="mesh-resolution"
                                                type="range"
                                                min="4"
                                                max="128"
                                                step="1"
                                                value={meshResolution}
                                                onChange={(e) => setMeshResolution(parseInt(e.target.value, 10))}
                                                disabled={selectedMesh === 'cube'}
                                                aria-label="Mesh resolution"
                                            />
                                            <span style={{width: '35px', textAlign: 'right'}}>{meshResolution}</span>
                                        </div>
                                        <div className="form-group toggle-group">
                                            <label htmlFor="wireframe-toggle">Show Wireframe</label>
                                            <label className="switch">
                                                <input
                                                    id="wireframe-toggle"
                                                    type="checkbox"
                                                    checked={showWireframe}
                                                    onChange={(e) => setShowWireframe(e.target.checked)}
                                                    aria-checked={showWireframe}
                                                />
                                                <span className="slider round"></span>
                                            </label>
                                        </div>
                                        <div className="control-divider"></div>
                                        <h3 className="control-subtitle">Material</h3>
                                        <div className="form-group">
                                            <label htmlFor="preset-select">Material Presets</label>
                                            <select 
                                                id="preset-select" 
                                                value={selectedPreset} 
                                                onChange={(e) => handlePresetChange(e.target.value)}
                                                aria-label="Select a material preset"
                                            >
                                                <option value="">-- Custom --</option>
                                                {SHADER_PRESETS.map(p => <option key={p.name} value={p.name}>{p.name}</option>)}
                                            </select>
                                        </div>
                                        <div className="form-group form-group-row">
                                            <label htmlFor="material-albedo">Albedo</label>
                                            <input
                                                id="material-albedo"
                                                type="color"
                                                value={materialState.albedo}
                                                onChange={(e) => setMaterialState(prev => ({ ...prev, albedo: e.target.value }))}
                                            />
                                        </div>
                                        <div className="form-group form-group-row">
                                            <label htmlFor="material-metallic">Metallic</label>
                                            <input
                                                id="material-metallic"
                                                type="range"
                                                min="0" max="1" step="0.01"
                                                value={materialState.metallic}
                                                onChange={(e) => setMaterialState(prev => ({ ...prev, metallic: parseFloat(e.target.value) }))}
                                            />
                                            <span>{materialState.metallic.toFixed(2)}</span>
                                        </div>
                                        <div className="form-group form-group-row">
                                            <label htmlFor="material-roughness">Roughness</label>
                                            <input
                                                id="material-roughness"
                                                type="range"
                                                min="0" max="1" step="0.01"
                                                value={materialState.roughness}
                                                onChange={(e) => setMaterialState(prev => ({ ...prev, roughness: parseFloat(e.target.value) }))}
                                            />
                                            <span>{materialState.roughness.toFixed(2)}</span>
                                        </div>
                                        <div className="control-divider"></div>
                                        <h3 className="control-subtitle">Lighting</h3>
                                        <div className="form-group">
                                            <label htmlFor="light-type-select">Light Type</label>
                                            <select 
                                                id="light-type-select" 
                                                value={lightState.type} 
                                                onChange={(e) => setLightState(prev => ({ ...prev, type: e.target.value }))}
                                                aria-label="Select light type"
                                            >
                                                <option value="hemispheric">Hemispheric</option>
                                                <option value="directional">Directional</option>
                                                <option value="point">Point</option>
                                            </select>
                                        </div>
                                        <div className="form-group form-group-row">
                                            <label htmlFor="light-intensity">Intensity</label>
                                            <input
                                                id="light-intensity"
                                                type="range"
                                                min="0"
                                                max="2"
                                                step="0.05"
                                                value={lightState.intensity}
                                                onChange={(e) => setLightState(prev => ({ ...prev, intensity: parseFloat(e.target.value) }))}
                                            />
                                            <span>{lightState.intensity.toFixed(2)}</span>
                                        </div>
                                        <div className="form-group form-group-row">
                                            <label htmlFor="light-color">Color</label>
                                            <input
                                                id="light-color"
                                                type="color"
                                                value={lightState.diffuse}
                                                onChange={(e) => setLightState(prev => ({ ...prev, diffuse: e.target.value }))}
                                            />
                                        </div>
                                        <div className="form-group">
                                            <label>{lightState.type === 'point' ? 'Position' : 'Direction'}</label>
                                            <div className="vector-inputs">
                                                <div className="vector-input-wrap">
                                                    <label htmlFor="light-dir-x">X</label>
                                                    <input
                                                        id="light-dir-x"
                                                        type="number"
                                                        step="0.1"
                                                        value={lightState.direction.x}
                                                        onChange={(e) => setLightState(prev => ({ ...prev, direction: { ...prev.direction, x: parseFloat(e.target.value) || 0 } }))}
                                                        aria-label={`Light ${lightState.type === 'point' ? 'position' : 'direction'} X`}
                                                    />
                                                </div>
                                                <div className="vector-input-wrap">
                                                    <label htmlFor="light-dir-y">Y</label>
                                                    <input
                                                        id="light-dir-y"
                                                        type="number"
                                                        step="0.1"
                                                        value={lightState.direction.y}
                                                        onChange={(e) => setLightState(prev => ({ ...prev, direction: { ...prev.direction, y: parseFloat(e.target.value) || 0 } }))}
                                                        aria-label={`Light ${lightState.type === 'point' ? 'position' : 'direction'} Y`}
                                                    />
                                                </div>
                                                <div className="vector-input-wrap">
                                                    <label htmlFor="light-dir-z">Z</label>
                                                     <input
                                                        id="light-dir-z"
                                                        type="number"
                                                        step="0.1"
                                                        value={lightState.direction.z}
                                                        onChange={(e) => setLightState(prev => ({ ...prev, direction: { ...prev.direction, z: parseFloat(e.target.value) || 0 } }))}
                                                        aria-label={`Light ${lightState.type === 'point' ? 'position' : 'direction'} Z`}
                                                    />
                                                </div>
                                            </div>
                                        </div>
                                        <div className="control-divider"></div>
                                        <h3 className="control-subtitle">Environment</h3>
                                        <div className="form-group">
                                            <div className="button-group">
                                                <label htmlFor="env-file-input" className="file-upload-button">
                                                    Upload
                                                </label>
                                                <input
                                                    id="env-file-input"
                                                    type="file"
                                                    accept="image/*,.hdr,.env"
                                                    onChange={handleFileChange}
                                                    style={{ display: 'none' }}
                                                />
                                                <button onClick={handleRandomBackground} className="button-secondary">Random</button>
                                                {environmentTexture && (
                                                    <button onClick={clearEnvironment} className="button-danger">Clear</button>
                                                )}
                                            </div>
                                        </div>
                                        <div className="control-divider"></div>
                                        <h3 className="control-subtitle">Post-Processing</h3>
                                        {/* Bloom Controls */}
                                        <div className="form-group toggle-group">
                                            <label htmlFor="pp-bloom-toggle">Bloom</label>
                                            <label className="switch">
                                                <input id="pp-bloom-toggle" type="checkbox" checked={postProcessingState.bloom.enabled} onChange={(e) => setPostProcessingState(p => ({ ...p, bloom: { ...p.bloom, enabled: e.target.checked } }))} />
                                                <span className="slider round"></span>
                                            </label>
                                        </div>
                                        <div className={`effect-options ${postProcessingState.bloom.enabled ? 'visible' : ''}`}>
                                            <div className="form-group form-group-row">
                                                <label htmlFor="pp-bloom-threshold">Threshold</label>
                                                <input id="pp-bloom-threshold" type="range" min="0" max="1" step="0.01" value={postProcessingState.bloom.threshold} onChange={(e) => setPostProcessingState(p => ({ ...p, bloom: { ...p.bloom, threshold: parseFloat(e.target.value) } }))} />
                                                <span>{postProcessingState.bloom.threshold.toFixed(2)}</span>
                                            </div>
                                            <div className="form-group form-group-row">
                                                <label htmlFor="pp-bloom-weight">Weight</label>
                                                <input id="pp-bloom-weight" type="range" min="0" max="1" step="0.01" value={postProcessingState.bloom.weight} onChange={(e) => setPostProcessingState(p => ({ ...p, bloom: { ...p.bloom, weight: parseFloat(e.target.value) } }))} />
                                                <span>{postProcessingState.bloom.weight.toFixed(2)}</span>
                                            </div>
                                             <div className="form-group form-group-row">
                                                <label htmlFor="pp-bloom-kernel">Size</label>
                                                <input id="pp-bloom-kernel" type="range" min="1" max="128" step="1" value={postProcessingState.bloom.kernel} onChange={(e) => setPostProcessingState(p => ({ ...p, bloom: { ...p.bloom, kernel: parseFloat(e.target.value) } }))} />
                                                <span>{postProcessingState.bloom.kernel.toFixed(0)}</span>
                                            </div>
                                        </div>
                                        {/* Other Effects */}
                                        <div className="form-group toggle-group">
                                            <label htmlFor="pp-fxaa-toggle">Anti-Aliasing</label>
                                            <label className="switch">
                                                <input id="pp-fxaa-toggle" type="checkbox" checked={postProcessingState.fxaa.enabled} onChange={(e) => setPostProcessingState(p => ({ ...p, fxaa: { enabled: e.target.checked } }))} />
                                                <span className="slider round"></span>
                                            </label>
                                        </div>
                                         <div className="form-group toggle-group">
                                            <label htmlFor="pp-chromatic-toggle">Chromatic Aberration</label>
                                            <label className="switch">
                                                <input id="pp-chromatic-toggle" type="checkbox" checked={postProcessingState.chromaticAberration.enabled} onChange={(e) => setPostProcessingState(p => ({ ...p, chromaticAberration: { ...p.chromaticAberration, enabled: e.target.checked } }))} />
                                                <span className="slider round"></span>
                                            </label>
                                        </div>
                                        <div className={`effect-options ${postProcessingState.chromaticAberration.enabled ? 'visible' : ''}`}>
                                            <div className="form-group form-group-row">
                                                <label htmlFor="pp-chromatic-amount">Amount</label>
                                                <input id="pp-chromatic-amount" type="range" min="-100" max="100" step="1" value={postProcessingState.chromaticAberration.aberrationAmount} onChange={(e) => setPostProcessingState(p => ({ ...p, chromaticAberration: { ...p.chromaticAberration, aberrationAmount: parseFloat(e.target.value) } }))} />
                                                <span>{postProcessingState.chromaticAberration.aberrationAmount.toFixed(0)}</span>
                                            </div>
                                        </div>
                                        <div className="form-group toggle-group">
                                            <label htmlFor="pp-grain-toggle">Film Grain</label>
                                            <label className="switch">
                                                <input id="pp-grain-toggle" type="checkbox" checked={postProcessingState.grain.enabled} onChange={(e) => setPostProcessingState(p => ({ ...p, grain: { ...p.grain, enabled: e.target.checked } }))} />
                                                <span className="slider round"></span>
                                            </label>
                                        </div>
                                        <div className={`effect-options ${postProcessingState.grain.enabled ? 'visible' : ''}`}>
                                            <div className="form-group form-group-row">
                                                <label htmlFor="pp-grain-intensity">Intensity</label>
                                                <input id="pp-grain-intensity" type="range" min="0" max="50" step="1" value={postProcessingState.grain.intensity} onChange={(e) => setPostProcessingState(p => ({ ...p, grain: { ...p.grain, intensity: parseFloat(e.target.value) } }))} />
                                                <span>{postProcessingState.grain.intensity.toFixed(0)}</span>
                                            </div>
                                        </div>
                                    </>
                                );
                                break;
                            case 'project':
                                title = 'Project Management';
                                content = (
                                    <>
                                        <div className="form-group">
                                            <label htmlFor="shader-name-input">Shader Name</label>
                                            <input
                                                id="shader-name-input"
                                                type="text"
                                                value={shaderName}
                                                onChange={(e) => setShaderName(e.target.value)}
                                                placeholder="My Awesome Shader"
                                            />
                                        </div>
                                        <button onClick={handleSaveShader} disabled={!shaderName.trim()}>Save Shader</button>
                                        
                                        <div className="form-group">
                                            <label htmlFor="load-shader-select">Load Shader</label>
                                            <select 
                                                id="load-shader-select"
                                                value={selectedShader}
                                                onChange={(e) => handleLoadShader(e.target.value)}
                                                aria-label="Select a saved shader to load"
                                            >
                                                <option value="">-- Select a Shader --</option>
                                                {savedShaders.map(shader => (
                                                    <option key={shader.name} value={shader.name}>{shader.name}</option>
                                                ))}
                                            </select>
                                        </div>
                                        <div className="button-group">
                                            <label htmlFor="import-shader-input" className="file-upload-button">
                                                Import
                                            </label>
                                            <input
                                                id="import-shader-input"
                                                type="file"
                                                accept=".json"
                                                onChange={handleImportShader}
                                                style={{ display: 'none' }}
                                                aria-label="Import shader from a JSON file"
                                            />
                                            <button onClick={handleExportShader} disabled={!selectedShader} className="button-secondary">
                                                Export
                                            </button>
                                            <button onClick={handleDeleteShader} disabled={!selectedShader} className="button-danger">
                                                Delete
                                            </button>
                                        </div>
                                    </>
                                );
                                break;
                            default:
                                return null;
                        }

                        const isCollapsed = !!collapsedPanels[key];

                        return (
                            <div
                                key={key}
                                className={`collapsible-panel ${!isCollapsed ? 'is-expanded' : ''}`}
                                onDragEnter={(e) => handleDragEnter(e, index)}
                                onDrop={handleDrop}
                                onDragOver={(e) => e.preventDefault()}
                            >
                                <h2 className={`panel-title collapsible ${isCollapsed ? 'is-collapsed' : 'is-expanded'}`} onClick={() => togglePanel(key)}>
                                    <span 
                                        className="drag-handle"
                                        draggable
                                        onDragStart={(e) => {
                                            e.stopPropagation();
                                            handleDragStart(e, index);
                                        }}
                                        onDragEnd={handleDragEnd}
                                        onClick={(e) => e.stopPropagation()}
                                        aria-label={`Drag to reorder ${title}`}
                                    >
                                        <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor" xmlns="http://www.w3.org/2000/svg"><path d="M11 18c0 1.1-.9 2-2 2s-2-.9-2-2 .9-2 2-2 2 .9 2 2zm-2-8c-1.1 0-2 .9-2 2s.9 2 2 2 2-.9 2-2-.9-2-2-2zm0-6c-1.1 0-2 .9-2 2s.9 2 2 2 2-.9 2-2-.9-2-2-2zm6 4c1.1 0 2-.9 2-2s-.9-2-2-2-2 .9-2 2 .9 2 2 2zm0 2c-1.1 0-2 .9-2 2s.9 2 2 2 2-.9 2-2-.9-2-2-2zm0 6c-1.1 0-2 .9-2 2s.9 2 2 2 2-.9 2-2-.9-2-2-2z"/></svg>
                                    </span>
                                    <span className="title-text">{title}</span>
                                    <svg className={`chevron ${isCollapsed ? 'collapsed' : ''}`} width="24" height="24" viewBox="0 0 24 24" fill="currentColor" xmlns="http://www.w3.org/2000/svg">
                                        <path d="M7.41 8.59L12 13.17l4.59-4.58L18 10l-6 6-6-6 1.41-1.41z"/>
                                    </svg>
                                </h2>
                                <div className={`panel-content ${isCollapsed ? 'collapsed' : ''}`}>
                                    {content}
                                </div>
                            </div>
                        )
                    })}
                </section>

                <section className="panel viewport-panel" aria-label="3D Viewport">
                    <canvas id="babylon-canvas" ref={babylonCanvas} touch-action="none" />
                    <div className="time-control-panel">
                        <button onClick={handleTogglePlay} className="time-control-button" aria-label={timeState.playing ? 'Pause' : 'Play'}>
                            {timeState.playing ? (
                                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor"><path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"/></svg>
                            ) : (
                                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>
                            )}
                        </button>
                        <button onClick={handleResetTime} className="time-control-button" aria-label="Reset Time">
                             <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor"><path d="M6 6h2v12H6zm3.5 6l8.5 6V6z"/></svg>
                        </button>
                        <span className="time-display">{timeState.time.toFixed(2)}s</span>
                        <input
                            type="range"
                            min="0"
                            max="60"
                            step="0.01"
                            value={timeState.time}
                            onChange={handleTimeScrub}
                            className="time-slider"
                            aria-label="Time Slider"
                        />
                    </div>
                </section>

                <section className="panel editor-panel" aria-labelledby="editor-title">
                    <div className="editor-header">
                        <div className="editor-tabs">
                            <button 
                                className={`tab-button ${activeTab === 'vertex' ? 'active' : ''}`}
                                onClick={() => setActiveTab('vertex')}
                                aria-pressed={activeTab === 'vertex'}>
                                Vertex Shader
                            </button>
                            <button 
                                className={`tab-button ${activeTab === 'fragment' ? 'active' : ''}`}
                                onClick={() => setActiveTab('fragment')}
                                aria-pressed={activeTab === 'fragment'}>
                                Fragment Shader
                            </button>
                        </div>
                        <div className="editor-actions">
                            <button onClick={handleOpenRefineModal} className="button-format" disabled={!hasSelection || isRefining || isLoading} aria-label="Refine selected code with AI">
                                Refine with AI
                            </button>
                             <button onClick={handleFormatCode} className="button-format" aria-label="Format active shader code">
                                Format Code
                            </button>
                        </div>
                    </div>
                    <div className="editor-content">
                        <div
                            ref={vertexEditorContainer}
                            style={{ display: activeTab === 'vertex' ? 'block' : 'none', height: '100%' }}
                            aria-label="Vertex Shader Code Editor"
                        />
                        <div
                            ref={fragmentEditorContainer}
                            style={{ display: activeTab === 'fragment' ? 'block' : 'none', height: '100%' }}
                            aria-label="Fragment Shader Code Editor"
                        />
                    </div>
                </section>
            </main>
            <footer className="app-footer" role="log" aria-live="assertive">
                {error || "No errors."}
            </footer>

            {refineModalOpen && (
                <div className="refine-modal-backdrop" onClick={closeRefineModal}>
                    <div className="refine-modal" onClick={(e) => e.stopPropagation()}>
                        <h3>Refine Code with AI</h3>
                        <div className="refine-modal-content">
                            <div className="form-group">
                                <label>Selected Code:</label>
                                <pre className="code-snippet"><code>{refinementSelection?.code}</code></pre>
                            </div>
                            <div className="form-group">
                                <label htmlFor="refine-prompt-input">How should the AI change it?</label>
                                <textarea
                                    id="refine-prompt-input"
                                    value={refinementPrompt}
                                    onChange={(e) => setRefinementPrompt(e.target.value)}
                                    placeholder="e.g., 'make this pulse slower' or 'change the color to a fiery orange'"
                                    aria-label="Enter your code refinement instructions here"
                                />
                            </div>
                        </div>
                        <div className="refine-modal-actions">
                            <button onClick={closeRefineModal} className="button-secondary" disabled={isRefining}>Cancel</button>
                            <button onClick={handleRefineCode} disabled={!refinementPrompt || isRefining}>
                                {isRefining ? <span className="loader" /> : 'Refine'}
                            </button>
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