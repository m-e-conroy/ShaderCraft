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
varying vec3 vNormal; // World-space normal from vertex shader
varying vec3 vPositionW; // World-space position from vertex shader

// Uniforms
uniform float u_time;

// Lighting Uniforms
uniform vec3 u_lightColor;
uniform float u_lightIntensity;
uniform int u_lightType; // 0: directional/hemispheric, 1: point
uniform vec3 u_lightDirection; // For directional/hemispheric
uniform vec3 u_lightPosition;  // For point light

// Environment/Reflection Uniforms
uniform samplerCube u_envTexture;
uniform vec3 u_cameraPosition;
uniform int u_hasEnvTexture; // Use int as a boolean (0 or 1)


void main(void) {
    // Base object color pattern
    vec3 objectColor = vec3(0.6 + 0.4 * sin(vUV.x * 20.0 + u_time), 0.6 + 0.4 * cos(vUV.y * 20.0 + u_time), 1.0);
    
    // --- Lighting Calculation ---
    vec3 ambientColor = vec3(0.15, 0.15, 0.15);
    
    vec3 lightDir;
    if (u_lightType == 0) { // Directional or Hemispheric
        lightDir = normalize(u_lightDirection);
    } else { // Point Light
        lightDir = normalize(u_lightPosition - vPositionW);
    }

    float diffuseFactor = max(0.0, dot(vNormal, lightDir));
    vec3 diffuseColor = diffuseFactor * u_lightColor * u_lightIntensity;
    vec3 litColor = objectColor * (ambientColor + diffuseColor);
    
    vec3 finalColor = litColor;

    // --- Reflection Calculation ---
    if (u_hasEnvTexture == 1) {
        vec3 viewDir = normalize(vPositionW - u_cameraPosition);
        // Use reflect on the view direction (from surface to eye)
        vec3 reflectDir = reflect(viewDir, normalize(vNormal));
        vec4 reflectionColor = textureCube(u_envTexture, reflectDir);
        
        // Mix the lit color with the reflection color.
        // The mix factor (0.75) is now controlled directly in the shader code.
        finalColor = mix(litColor, reflectionColor.rgb, 0.75);
    }

    gl_FragColor = vec4(finalColor, 1.0);
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
    {
        name: 'Basic Lighting',
        vertex: DEFAULT_VERTEX_SHADER,
        fragment: `
precision highp float;
varying vec2 vUV;
varying vec3 vNormal;
varying vec3 vPositionW;
uniform vec3 u_lightColor;
uniform float u_lightIntensity;
uniform int u_lightType;
uniform vec3 u_lightDirection;
uniform vec3 u_lightPosition;

void main(void) {
    vec3 objectColor = vec3(0.8, 0.2, 0.2); // Simple red color
    vec3 ambient = vec3(0.1);
    vec3 normal = normalize(vNormal);

    vec3 lightDir;
    if (u_lightType == 0) { // Directional/Hemispheric
        lightDir = normalize(u_lightDirection);
    } else { // Point Light
        lightDir = normalize(u_lightPosition - vPositionW);
    }

    float diffuseFactor = max(dot(normal, lightDir), 0.0);
    vec3 diffuse = diffuseFactor * u_lightColor * u_lightIntensity;
    vec3 finalColor = objectColor * (ambient + diffuse);

    gl_FragColor = vec4(finalColor, 1.0);
}`.trim(),
    },
    {
        name: 'Cel Shading',
        vertex: DEFAULT_VERTEX_SHADER,
        fragment: `
precision highp float;
varying vec2 vUV;
varying vec3 vNormal;
varying vec3 vPositionW;
uniform vec3 u_lightColor;
uniform float u_lightIntensity;
uniform int u_lightType;
uniform vec3 u_lightDirection;
uniform vec3 u_lightPosition;

void main(void) {
    vec3 objectColor = vec3(0.2, 0.6, 0.9);
    vec3 normal = normalize(vNormal);

    vec3 lightDir;
    if (u_lightType == 0) {
        lightDir = normalize(u_lightDirection);
    } else {
        lightDir = normalize(u_lightPosition - vPositionW);
    }

    float diffuse = max(0.0, dot(normal, lightDir)) * u_lightIntensity;

    // Create hard steps for the cartoon effect
    float celValue;
    if (diffuse > 0.95) {
        celValue = 1.0;
    } else if (diffuse > 0.6) {
        celValue = 0.7;
    } else if (diffuse > 0.2) {
        celValue = 0.4;
    } else {
        celValue = 0.2;
    }

    vec3 finalColor = objectColor * celValue * u_lightColor;
    gl_FragColor = vec4(finalColor, 1.0);
}`.trim(),
    },
    {
        name: 'Glossy Plastic',
        vertex: DEFAULT_VERTEX_SHADER,
        fragment: `
precision highp float;
varying vec2 vUV;
varying vec3 vNormal;
varying vec3 vPositionW;
uniform vec3 u_cameraPosition;
uniform vec3 u_lightColor;
uniform float u_lightIntensity;
uniform int u_lightType;
uniform vec3 u_lightDirection;
uniform vec3 u_lightPosition;

void main(void) {
    vec3 objectColor = vec3(0.1, 0.8, 0.3); // Green plastic
    float shininess = 32.0;

    vec3 ambient = vec3(0.1);
    vec3 normal = normalize(vNormal);
    vec3 viewDir = normalize(u_cameraPosition - vPositionW);

    vec3 lightDir;
    if (u_lightType == 0) {
        lightDir = normalize(u_lightDirection);
    } else {
        lightDir = normalize(u_lightPosition - vPositionW);
    }
    
    // Diffuse
    float diffuseFactor = max(dot(normal, lightDir), 0.0);
    vec3 diffuse = diffuseFactor * u_lightColor;

    // Specular (Blinn-Phong)
    vec3 halfwayDir = normalize(lightDir + viewDir);
    float specAngle = max(dot(normal, halfwayDir), 0.0);
    float specularFactor = pow(specAngle, shininess);
    vec3 specular = specularFactor * u_lightColor;

    vec3 finalColor = objectColor * (ambient + diffuse) * u_lightIntensity + specular * u_lightIntensity;

    gl_FragColor = vec4(finalColor, 1.0);
}`.trim(),
    },
    {
        name: 'Metallic Reflection',
        vertex: DEFAULT_VERTEX_SHADER,
        fragment: `
precision highp float;
varying vec3 vNormal;
varying vec3 vPositionW;
uniform vec3 u_cameraPosition;
uniform samplerCube u_envTexture;
uniform int u_hasEnvTexture;

void main(void) {
    if (u_hasEnvTexture == 0) {
        // Fallback color if no environment map is present
        gl_FragColor = vec4(0.8, 0.8, 0.8, 1.0);
        return;
    }

    vec3 viewDir = normalize(vPositionW - u_cameraPosition);
    vec3 normal = normalize(vNormal);
    vec3 reflectDir = reflect(viewDir, normal);
    
    // Sample the environment cube map for the reflection
    vec4 reflectionColor = textureCube(u_envTexture, reflectDir);

    gl_FragColor = vec4(reflectionColor.rgb, 1.0);
}`.trim(),
    },
    {
        name: 'Psychedelic Wobble',
        vertex: `
precision highp float;
attribute vec3 position;
attribute vec2 uv;
attribute vec3 normal;
uniform mat4 worldViewProjection;
uniform mat4 world;
uniform float u_time;

varying vec2 vUV;
varying vec3 vNormal;
varying vec3 vPositionW;

void main(void) {
    // Animate vertex positions
    float wobbleFactor = 0.2 * sin(position.y * 5.0 + u_time * 2.0);
    vec3 wobbledPosition = position + normal * wobbleFactor;

    vec4 worldPosition = world * vec4(wobbledPosition, 1.0);
    gl_Position = worldViewProjection * vec4(wobbledPosition, 1.0);
    
    vUV = uv;
    vPositionW = worldPosition.xyz;
    
    // Transform normal to world space and normalize it.
    // Note: for perfect accuracy with the wobbled position, the normal vector itself should be recalculated.
    vec3 worldNormal = mat3(world) * normal;
    vNormal = normalize(worldNormal);
}`.trim(),
        fragment: `
precision highp float;
varying vec2 vUV;
varying vec3 vNormal;
uniform float u_time;
uniform vec3 u_lightColor;
uniform float u_lightIntensity;
uniform vec3 u_lightDirection;

void main(void) {
    // Create a colorful pattern that changes over time
    float r = 0.5 + 0.5 * sin(vUV.x * 10.0 + u_time);
    float g = 0.5 + 0.5 * cos(vUV.y * 10.0 + u_time * 1.5);
    float b = 0.5 + 0.5 * sin(vNormal.z * 5.0 + u_time * 0.5);
    vec3 objectColor = vec3(r, g, b);

    // Simple lighting
    vec3 normal = normalize(vNormal);
    float diffuse = max(0.0, dot(normal, normalize(u_lightDirection)));
    
    gl_FragColor = vec4(objectColor * (0.3 + diffuse * u_lightIntensity), 1.0);
}`.trim(),
    }
];

interface SavedShader {
    name: string;
    vertex: string;
    fragment: string;
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

// FIX: Removed `: React.FC` to simplify the component definition and avoid potential complex type-checking issues.
const App = () => {
    const [vertexCode, setVertexCode] = useState<string>(DEFAULT_VERTEX_SHADER);
    const [fragmentCode, setFragmentCode] = useState<string>(DEFAULT_FRAGMENT_SHADER);
    const [prompt, setPrompt] = useState<string>('');
    const [isLoading, setIsLoading] = useState<boolean>(false);
    const [promptActionLoading, setPromptActionLoading] = useState<'random' | 'enhance' | null>(null);
    const [error, setError] = useState<string>('');
    const [activeTab, setActiveTab] = useState<'vertex' | 'fragment'>('vertex');
    const [selectedMesh, setSelectedMesh] = useState<string>('sphere');
    const [meshResolution, setMeshResolution] = useState<number>(32);
    const [showWireframe, setShowWireframe] = useState<boolean>(false);
    const [lightState, setLightState] = useState({
        type: 'hemispheric',
        intensity: 1.0,
        diffuse: '#ffffff',
        direction: { x: 1, y: 1, z: 0 } // Re-used for position in point lights
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
    const [llmProvider, setLlmProvider] = useState<'gemini' | 'local'>(() => getInitialState('shadercraft_llm_provider', 'gemini'));
    const [localLlmEndpoint, setLocalLlmEndpoint] = useState<string>(() => getInitialState('shadercraft_llm_endpoint', 'http://localhost:11434/api/generate'));
    const [localLlmModel, setLocalLlmModel] = useState<string>(() => getInitialState('shadercraft_llm_model', 'codellama'));
    const [localLlmStatus, setLocalLlmStatus] = useState<'unchecked' | 'connected' | 'error'>('unchecked');

    // State for AI Refinement
    const [hasSelection, setHasSelection] = useState<boolean>(false);
    const [isRefining, setIsRefining] = useState<boolean>(false);
    const [refineModalOpen, setRefineModalOpen] = useState<boolean>(false);
    const [refinementPrompt, setRefinementPrompt] = useState<string>('');
    const [refinementSelection, setRefinementSelection] = useState<RefinementSelection | null>(null);

    const babylonCanvas = useRef<HTMLCanvasElement | null>(null);
    const sceneRef = useRef<any>(null);
    const engineRef = useRef<any>(null);
    const meshRef = useRef<any>(null);
    const lightRef = useRef<any>(null);
    const skyboxRef = useRef<any>(null);
    const ppPipelineRef = useRef<any>(null);
    const lightStateRef = useRef(lightState);
    const prevSelectedMeshRef = useRef<string | undefined>(undefined);

    const vertexEditorContainer = useRef<HTMLDivElement | null>(null);
    const fragmentEditorContainer = useRef<HTMLDivElement | null>(null);
    const vertexCmRef = useRef<any>(null);
    const fragmentCmRef = useRef<any>(null);

    const dragItem = useRef<number | null>(null);
    const dragOverItem = useRef<number | null>(null);

    // Keep refs in sync with the latest state to avoid stale closures in the render loop
    useEffect(() => {
        lightStateRef.current = lightState;
    }, [lightState]);

    // Save LLM settings to localStorage
    useEffect(() => {
        localStorage.setItem('shadercraft_llm_provider', JSON.stringify(llmProvider));
        localStorage.setItem('shadercraft_llm_endpoint', JSON.stringify(localLlmEndpoint));
        localStorage.setItem('shadercraft_llm_model', JSON.stringify(localLlmModel));
    }, [llmProvider, localLlmEndpoint, localLlmModel]);

    // Test local LLM connection
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

    // Load saved shaders from localStorage on initial mount
    useEffect(() => {
        setSavedShaders(getInitialState('shadercraft_shaders', []));
    }, []);

    const handleSaveShader = () => {
        if (!shaderName.trim()) {
            alert("Please enter a name for your shader.");
            return;
        }

        const newShader = { name: shaderName.trim(), vertex: vertexCode, fragment: fragmentCode };
        
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
                    "u_lightPosition", "u_lightType", "u_cameraPosition", "u_hasEnvTexture"
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

        let time = 0;
        engine.runRenderLoop(() => {
            const material = scene.getMaterialByName("customShader");
            if (material && material.getClassName() === "ShaderMaterial") {
                const ls = lightStateRef.current;
                const lightVector = new BABYLON.Vector3(ls.direction.x, ls.direction.y, ls.direction.z);
                
                (material as any).setFloat("u_time", time);
                material.setFloat("u_lightIntensity", ls.intensity);
                material.setColor3("u_lightColor", BABYLON.Color3.FromHexString(ls.diffuse));

                if (ls.type === 'point') {
                    material.setInt("u_lightType", 1);
                    material.setVector3("u_lightPosition", lightVector);
                } else { // Hemispheric and Directional
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

                time += engine.getDeltaTime() / 1000;
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
        const setupEditor = (container: HTMLElement | null, value: string, mode: string, cmRef: React.MutableRefObject<any>, setCode: (code: string) => void) => {
            if (container && !cmRef.current) {
                const cm = CodeMirror(container, {
                    value: value,
                    mode: mode,
                    theme: 'material-darker',
                    lineNumbers: true,
                });
                cm.on('change', (instance: any) => setCode(instance.getValue()));
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

        setupEditor(vertexEditorContainer.current, vertexCode, 'x-shader/x-vertex', vertexCmRef, setVertexCode);
        setupEditor(fragmentEditorContainer.current, fragmentCode, 'x-shader/x-fragment', fragmentCmRef, setFragmentCode);

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
    
    // Extracts a JSON block from a string that might be wrapped in markdown
    const extractJsonFromString = (str: string): string | null => {
        const match = str.match(/```json\s*([\s\S]*?)\s*```|({[\s\S]*})/);
        return match ? (match[1] || match[2]) : null;
    };


    const handleGenerateShader = async () => {
        setIsLoading(true);
        setError('');

        const isRefinement = vertexCode !== DEFAULT_VERTEX_SHADER || fragmentCode !== DEFAULT_FRAGMENT_SHADER;

        // FIX: Separated system instruction from user content for better prompting.
        const systemInstruction = `You are an expert in GLSL and Babylon.js. Create GLSL shaders that will run within a Babylon.js ShaderMaterial.

Please provide the complete GLSL code for both the vertex and fragment shaders.

- The vertex shader MUST define \`gl_Position\`.
- It will receive attributes: \`vec3 position\`, \`vec3 normal\`, \`vec2 uv\`.
- It MUST pass a varying \`vUV\` (\`vec2\`), \`vNormal\` (\`vec3\`), and \`vPositionW\` (\`vec3\`) to the fragment shader.
- The fragment shader MUST define \`gl_FragColor\`.
- It will receive the varyings \`vUV\`, \`vNormal\`, and \`vPositionW\`.
- Babylon.js provides these uniforms automatically: \`mat4 worldViewProjection\`, \`mat4 world\`, \`mat4 view\`, \`mat4 projection\`.
- Custom uniforms are also provided for animations and lighting: \`float u_time\`, \`vec3 u_lightColor\`, \`float u_lightIntensity\`, \`vec3 u_lightDirection\`, \`vec3 u_lightPosition\`, \`int u_lightType\`, \`vec3 u_cameraPosition\`.
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
            let jsonString: string | null = null;

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
                jsonString = response.text.trim();

            } else { // Local LLM provider
                if (localLlmStatus !== 'connected') {
                    throw new Error("Local LLM endpoint is not connected. Please check the URL.");
                }
                const response = await fetch(localLlmEndpoint, {
                    method: 'POST',
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
                const content = responseData.response || responseData.content || JSON.stringify(responseData);
                jsonString = extractJsonFromString(content) || content;
            }

            if (!jsonString) {
                throw new Error("AI response was empty or malformed.");
            }

            const shaderData = JSON.parse(jsonString);

            if (shaderData.vertexShader && shaderData.fragmentShader) {
                const formattedVertex = await formatGlslCode(shaderData.vertexShader);
                const formattedFragment = await formatGlslCode(shaderData.fragmentShader);
                setVertexCode(formattedVertex);
                setFragmentCode(formattedFragment);
                setSelectedPreset(''); // Clear preset selection after generating
            } else {
                setError("AI response was missing shader code. Please try again.");
            }

        } catch (e: any) {
            console.error(e);
            setError(e instanceof Error ? e.message : 'An unknown error occurred.');
        } finally {
            setIsLoading(false);
        }
    };

    const handlePromptAction = async (action: 'random' | 'enhance') => {
        setPromptActionLoading(action);
        setError('');

        const content = action === 'random' 
            ? "Generate a single, short, and creative prompt for a GLSL shader effect. Examples: 'liquid mercury flowing over a sphere', 'holographic glitch effect', 'a shield made of swirling energy'. Return only the prompt text itself, no extra words or formatting."
            : `You are a creative assistant for a 3D artist. Take the following shader idea and enhance it, making it more descriptive, vivid, and inspiring, but keep it as a concise prompt. User's idea: "${prompt}". Return only the enhanced prompt text, without any introductory phrases.`;
        
        try {
            let resultText: string;

            if (llmProvider === 'gemini') {
                const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
                const response = await ai.models.generateContent({
                    model: "gemini-2.5-flash",
                    contents: content,
                });
                resultText = response.text;
            } else { // Local LLM
                if (localLlmStatus !== 'connected') {
                    throw new Error("Local LLM endpoint is not connected.");
                }
                const response = await fetch(localLlmEndpoint, {
                     method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ model: localLlmModel, prompt: content, stream: false })
                });
                if (!response.ok) throw new Error(`Local LLM request failed: ${response.statusText}`);
                const data = await response.json();
                resultText = data.response || data.content || '';
            }

            setPrompt(resultText.trim().replace(/['"]+/g, '')); // Clean up quotes
        } catch (e: any) {
            console.error(e);
            setError(e instanceof Error ? e.message : `Failed to ${action} prompt.`);
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
            } else {
                if (localLlmStatus !== 'connected') {
                    throw new Error("Local LLM endpoint is not connected.");
                }
                 const response = await fetch(localLlmEndpoint, {
                     method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ model: localLlmModel, prompt: `${systemInstruction}\n\n${userContent}`, stream: false })
                });
                if (!response.ok) throw new Error(`Local LLM request failed: ${response.statusText}`);
                const data = await response.json();
                refinedCode = data.response || data.content || '';
            }

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
            console.error("Refinement failed:", e);
            setError(e instanceof Error ? e.message : "Failed to refine code.");
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
            setVertexCode(preset.vertex);
            setFragmentCode(preset.fragment);
            setSelectedShader(''); // Clear saved shader selection
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
                            case 'ai':
                                title = 'AI Controls';
                                content = (
                                    <>
                                        <div className="form-group">
                                            <label htmlFor="llm-provider-select">AI Provider</label>
                                            <select
                                                id="llm-provider-select"
                                                value={llmProvider}
                                                onChange={(e) => setLlmProvider(e.target.value as 'gemini' | 'local')}
                                            >
                                                <option value="gemini">Gemini API</option>
                                                <option value="local">Local LLM</option>
                                            </select>
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
                                        {llmProvider === 'gemini' && (
                                            <div className="form-group">
                                                <label htmlFor="preset-select">Shader Presets</label>
                                                <select 
                                                    id="preset-select" 
                                                    value={selectedPreset} 
                                                    onChange={(e) => handlePresetChange(e.target.value)}
                                                    aria-label="Select a shader preset"
                                                >
                                                    <option value="">-- Select a Preset --</option>
                                                    {SHADER_PRESETS.map(p => <option key={p.name} value={p.name}>{p.name}</option>)}
                                                </select>
                                            </div>
                                        )}
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