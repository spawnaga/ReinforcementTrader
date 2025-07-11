// Revolutionary 3D Portfolio Visualization using Three.js

class Portfolio3DVisualization {
    constructor(containerId) {
        this.containerId = containerId;
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        this.portfolioMesh = null;
        this.riskSurface = null;
        this.particleSystem = null;
        this.animationId = null;
        
        // Portfolio data
        this.portfolioData = {
            positions: [],
            riskMetrics: {},
            performance: {},
            marketData: []
        };
        
        // Animation parameters
        this.time = 0;
        this.rotationSpeed = 0.01;
        this.pulseSpeed = 0.02;
        
        console.log('🎮 3D Portfolio Visualization initialized');
    }
    
    init() {
        this.setupScene();
        this.setupCamera();
        this.setupRenderer();
        this.setupControls();
        this.setupLighting();
        this.createPortfolioVisualization();
        this.createRiskSurface();
        this.createParticleSystem();
        this.setupEventListeners();
        this.startAnimation();
        
        console.log('✅ 3D Visualization ready');
    }
    
    setupScene() {
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x0a0a0a);
        this.scene.fog = new THREE.Fog(0x0a0a0a, 50, 200);
    }
    
    setupCamera() {
        const container = document.getElementById(this.containerId);
        const width = container.clientWidth;
        const height = container.clientHeight;
        
        this.camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000);
        this.camera.position.set(0, 20, 40);
        this.camera.lookAt(0, 0, 0);
    }
    
    setupRenderer() {
        const container = document.getElementById(this.containerId);
        
        this.renderer = new THREE.WebGLRenderer({
            antialias: true,
            alpha: true,
            powerPreference: 'high-performance'
        });
        
        this.renderer.setSize(container.clientWidth, container.clientHeight);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        this.renderer.outputEncoding = THREE.sRGBEncoding;
        this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
        this.renderer.toneMappingExposure = 1.2;
        
        container.appendChild(this.renderer.domElement);
    }
    
    setupControls() {
        this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        this.controls.screenSpacePanning = false;
        this.controls.minDistance = 10;
        this.controls.maxDistance = 100;
        this.controls.maxPolarAngle = Math.PI / 2;
    }
    
    setupLighting() {
        // Ambient light
        const ambientLight = new THREE.AmbientLight(0x404040, 0.3);
        this.scene.add(ambientLight);
        
        // Main directional light
        const directionalLight = new THREE.DirectionalLight(0x00ff88, 1);
        directionalLight.position.set(20, 20, 20);
        directionalLight.castShadow = true;
        directionalLight.shadow.mapSize.width = 2048;
        directionalLight.shadow.mapSize.height = 2048;
        directionalLight.shadow.camera.near = 0.5;
        directionalLight.shadow.camera.far = 100;
        directionalLight.shadow.camera.left = -50;
        directionalLight.shadow.camera.right = 50;
        directionalLight.shadow.camera.top = 50;
        directionalLight.shadow.camera.bottom = -50;
        this.scene.add(directionalLight);
        
        // Accent lights
        const light1 = new THREE.PointLight(0x0088ff, 0.5, 50);
        light1.position.set(-20, 10, 10);
        this.scene.add(light1);
        
        const light2 = new THREE.PointLight(0xff0088, 0.5, 50);
        light2.position.set(20, 10, -10);
        this.scene.add(light2);
        
        // Holographic rim light
        const rimLight = new THREE.DirectionalLight(0x00ffff, 0.3);
        rimLight.position.set(-20, -20, -20);
        this.scene.add(rimLight);
    }
    
    createPortfolioVisualization() {
        // Create main portfolio sphere
        const geometry = new THREE.SphereGeometry(8, 64, 32);
        
        // Create holographic material
        const material = new THREE.ShaderMaterial({
            uniforms: {
                time: { value: 0 },
                colorA: { value: new THREE.Color(0x00ff88) },
                colorB: { value: new THREE.Color(0x0088ff) },
                colorC: { value: new THREE.Color(0xff0088) },
                opacity: { value: 0.8 }
            },
            vertexShader: `
                varying vec3 vPosition;
                varying vec3 vNormal;
                varying vec2 vUv;
                
                void main() {
                    vPosition = position;
                    vNormal = normal;
                    vUv = uv;
                    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
                }
            `,
            fragmentShader: `
                uniform float time;
                uniform vec3 colorA;
                uniform vec3 colorB;
                uniform vec3 colorC;
                uniform float opacity;
                
                varying vec3 vPosition;
                varying vec3 vNormal;
                varying vec2 vUv;
                
                void main() {
                    vec3 color = mix(colorA, colorB, sin(vPosition.y * 2.0 + time) * 0.5 + 0.5);
                    color = mix(color, colorC, sin(vPosition.x * 2.0 + time * 0.7) * 0.5 + 0.5);
                    
                    float fresnel = pow(1.0 - abs(dot(vNormal, vec3(0.0, 0.0, 1.0))), 2.0);
                    color = mix(color, vec3(1.0), fresnel * 0.3);
                    
                    float alpha = opacity * (0.7 + 0.3 * sin(time * 2.0));
                    
                    gl_FragColor = vec4(color, alpha);
                }
            `,
            transparent: true,
            blending: THREE.AdditiveBlending
        });
        
        this.portfolioMesh = new THREE.Mesh(geometry, material);
        this.portfolioMesh.position.set(0, 0, 0);
        this.scene.add(this.portfolioMesh);
        
        // Create wireframe overlay
        const wireframeGeometry = new THREE.SphereGeometry(8.1, 32, 16);
        const wireframeMaterial = new THREE.MeshBasicMaterial({
            color: 0x00ff88,
            wireframe: true,
            transparent: true,
            opacity: 0.3
        });
        
        const wireframeMesh = new THREE.Mesh(wireframeGeometry, wireframeMaterial);
        this.scene.add(wireframeMesh);
        
        // Create position indicators
        this.createPositionIndicators();
    }
    
    createPositionIndicators() {
        const positionGroup = new THREE.Group();
        
        // Create position cubes around the sphere
        for (let i = 0; i < 8; i++) {
            const angle = (i / 8) * Math.PI * 2;
            const radius = 12;
            
            const geometry = new THREE.BoxGeometry(1, 1, 1);
            const material = new THREE.MeshLambertMaterial({
                color: new THREE.Color().setHSL(i / 8, 0.8, 0.6),
                transparent: true,
                opacity: 0.8
            });
            
            const cube = new THREE.Mesh(geometry, material);
            cube.position.set(
                Math.cos(angle) * radius,
                Math.sin(i * 0.5) * 3,
                Math.sin(angle) * radius
            );
            
            cube.userData = {
                originalPosition: cube.position.clone(),
                index: i
            };
            
            positionGroup.add(cube);
        }
        
        this.scene.add(positionGroup);
        this.positionIndicators = positionGroup;
    }
    
    createRiskSurface() {
        // Create risk surface as a plane with displacement
        const geometry = new THREE.PlaneGeometry(40, 40, 64, 64);
        const material = new THREE.ShaderMaterial({
            uniforms: {
                time: { value: 0 },
                riskLevel: { value: 0.5 },
                colorLow: { value: new THREE.Color(0x00ff88) },
                colorHigh: { value: new THREE.Color(0xff0088) }
            },
            vertexShader: `
                uniform float time;
                uniform float riskLevel;
                
                varying vec2 vUv;
                varying float vElevation;
                
                void main() {
                    vUv = uv;
                    
                    vec3 pos = position;
                    float elevation = sin(pos.x * 0.3 + time) * sin(pos.y * 0.3 + time * 0.7) * riskLevel * 5.0;
                    pos.z += elevation;
                    
                    vElevation = elevation;
                    
                    gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
                }
            `,
            fragmentShader: `
                uniform vec3 colorLow;
                uniform vec3 colorHigh;
                
                varying vec2 vUv;
                varying float vElevation;
                
                void main() {
                    vec3 color = mix(colorLow, colorHigh, (vElevation + 2.5) / 5.0);
                    float alpha = 0.3 + abs(vElevation) * 0.1;
                    
                    gl_FragColor = vec4(color, alpha);
                }
            `,
            transparent: true,
            side: THREE.DoubleSide
        });
        
        this.riskSurface = new THREE.Mesh(geometry, material);
        this.riskSurface.rotation.x = -Math.PI / 2;
        this.riskSurface.position.y = -15;
        this.scene.add(this.riskSurface);
    }
    
    createParticleSystem() {
        const particleCount = 1000;
        const geometry = new THREE.BufferGeometry();
        const positions = new Float32Array(particleCount * 3);
        const colors = new Float32Array(particleCount * 3);
        const sizes = new Float32Array(particleCount);
        
        for (let i = 0; i < particleCount; i++) {
            // Random positions in a sphere
            const radius = Math.random() * 50 + 20;
            const theta = Math.random() * Math.PI * 2;
            const phi = Math.acos(2 * Math.random() - 1);
            
            positions[i * 3] = radius * Math.sin(phi) * Math.cos(theta);
            positions[i * 3 + 1] = radius * Math.sin(phi) * Math.sin(theta);
            positions[i * 3 + 2] = radius * Math.cos(phi);
            
            // Random colors
            const color = new THREE.Color();
            color.setHSL(Math.random() * 0.3 + 0.5, 0.8, 0.6);
            colors[i * 3] = color.r;
            colors[i * 3 + 1] = color.g;
            colors[i * 3 + 2] = color.b;
            
            sizes[i] = Math.random() * 2 + 1;
        }
        
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        geometry.setAttribute('size', new THREE.BufferAttribute(sizes, 1));
        
        const material = new THREE.ShaderMaterial({
            uniforms: {
                time: { value: 0 },
                pointTexture: { value: this.createParticleTexture() }
            },
            vertexShader: `
                attribute float size;
                uniform float time;
                
                varying vec3 vColor;
                
                void main() {
                    vColor = color;
                    
                    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
                    gl_PointSize = size * (300.0 / -mvPosition.z) * (0.5 + 0.5 * sin(time * 2.0));
                    gl_Position = projectionMatrix * mvPosition;
                }
            `,
            fragmentShader: `
                uniform sampler2D pointTexture;
                
                varying vec3 vColor;
                
                void main() {
                    vec4 texColor = texture2D(pointTexture, gl_PointCoord);
                    gl_FragColor = vec4(vColor * texColor.rgb, texColor.a);
                }
            `,
            vertexColors: true,
            transparent: true,
            blending: THREE.AdditiveBlending
        });
        
        this.particleSystem = new THREE.Points(geometry, material);
        this.scene.add(this.particleSystem);
    }
    
    createParticleTexture() {
        const canvas = document.createElement('canvas');
        canvas.width = 64;
        canvas.height = 64;
        
        const context = canvas.getContext('2d');
        const gradient = context.createRadialGradient(32, 32, 0, 32, 32, 32);
        gradient.addColorStop(0, 'rgba(255, 255, 255, 1)');
        gradient.addColorStop(0.2, 'rgba(0, 255, 136, 0.8)');
        gradient.addColorStop(0.4, 'rgba(0, 136, 255, 0.4)');
        gradient.addColorStop(1, 'rgba(0, 0, 0, 0)');
        
        context.fillStyle = gradient;
        context.fillRect(0, 0, 64, 64);
        
        const texture = new THREE.CanvasTexture(canvas);
        texture.needsUpdate = true;
        
        return texture;
    }
    
    setupEventListeners() {
        // Handle window resize
        window.addEventListener('resize', () => {
            this.handleResize();
        });
        
        // Handle mouse interactions
        this.renderer.domElement.addEventListener('click', (event) => {
            this.handleClick(event);
        });
        
        // Handle data updates
        document.addEventListener('portfolioDataUpdate', (event) => {
            this.updateVisualization(event.detail);
        });
    }
    
    handleResize() {
        const container = document.getElementById(this.containerId);
        const width = container.clientWidth;
        const height = container.clientHeight;
        
        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(width, height);
    }
    
    handleClick(event) {
        const rect = this.renderer.domElement.getBoundingClientRect();
        const mouse = new THREE.Vector2();
        mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
        mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
        
        const raycaster = new THREE.Raycaster();
        raycaster.setFromCamera(mouse, this.camera);
        
        const intersects = raycaster.intersectObjects(this.positionIndicators.children);
        
        if (intersects.length > 0) {
            const clickedObject = intersects[0].object;
            this.highlightPosition(clickedObject);
            
            // Emit event for position selection
            const event = new CustomEvent('positionSelected', {
                detail: { index: clickedObject.userData.index }
            });
            document.dispatchEvent(event);
        }
    }
    
    highlightPosition(object) {
        // Reset all positions
        this.positionIndicators.children.forEach(child => {
            child.material.opacity = 0.8;
            child.scale.setScalar(1);
        });
        
        // Highlight selected position
        object.material.opacity = 1;
        object.scale.setScalar(1.5);
        
        // Add pulsing animation
        const originalScale = object.scale.clone();
        const pulseTween = new TWEEN.Tween(object.scale)
            .to({ x: originalScale.x * 1.2, y: originalScale.y * 1.2, z: originalScale.z * 1.2 }, 500)
            .easing(TWEEN.Easing.Quadratic.InOut)
            .yoyo(true)
            .repeat(3);
        
        pulseTween.start();
    }
    
    updateVisualization(data) {
        this.portfolioData = { ...this.portfolioData, ...data };
        
        // Update portfolio sphere based on performance
        if (this.portfolioMesh && data.performance) {
            const performance = data.performance.totalReturn || 0;
            const scale = 1 + performance * 0.5;
            this.portfolioMesh.scale.setScalar(Math.max(0.5, Math.min(2, scale)));
            
            // Update material uniforms
            if (this.portfolioMesh.material.uniforms) {
                this.portfolioMesh.material.uniforms.opacity.value = 
                    0.8 + Math.abs(performance) * 0.2;
            }
        }
        
        // Update risk surface
        if (this.riskSurface && data.riskMetrics) {
            const riskLevel = data.riskMetrics.maxDrawdown || 0;
            this.riskSurface.material.uniforms.riskLevel.value = Math.abs(riskLevel);
        }
        
        // Update position indicators
        if (this.positionIndicators && data.positions) {
            data.positions.forEach((position, index) => {
                if (index < this.positionIndicators.children.length) {
                    const indicator = this.positionIndicators.children[index];
                    
                    // Update color based on P&L
                    const pnl = position.unrealizedPnl || 0;
                    const color = pnl > 0 ? 
                        new THREE.Color(0x00ff88) : 
                        new THREE.Color(0xff0088);
                    
                    indicator.material.color = color;
                    
                    // Update size based on position size
                    const size = Math.abs(position.size || 1);
                    indicator.scale.setScalar(Math.max(0.5, Math.min(2, size)));
                }
            });
        }
    }
    
    updatePrice(price) {
        // Update visualization based on price changes
        if (this.portfolioMesh) {
            const priceChange = (price - (this.lastPrice || price)) / price;
            this.lastPrice = price;
            
            // Pulse effect based on price change
            const intensity = Math.abs(priceChange) * 1000;
            const color = priceChange > 0 ? 
                new THREE.Color(0x00ff88) : 
                new THREE.Color(0xff0088);
            
            // Update lighting
            if (this.scene.children) {
                this.scene.children.forEach(child => {
                    if (child instanceof THREE.PointLight) {
                        child.color = color;
                        child.intensity = 0.5 + intensity;
                    }
                });
            }
        }
    }
    
    startAnimation() {
        const animate = () => {
            this.animationId = requestAnimationFrame(animate);
            
            this.time += 0.016; // ~60fps
            
            // Update controls
            this.controls.update();
            
            // Update shader uniforms
            if (this.portfolioMesh && this.portfolioMesh.material.uniforms) {
                this.portfolioMesh.material.uniforms.time.value = this.time;
            }
            
            if (this.riskSurface && this.riskSurface.material.uniforms) {
                this.riskSurface.material.uniforms.time.value = this.time;
            }
            
            if (this.particleSystem && this.particleSystem.material.uniforms) {
                this.particleSystem.material.uniforms.time.value = this.time;
            }
            
            // Rotate portfolio sphere
            if (this.portfolioMesh) {
                this.portfolioMesh.rotation.y += this.rotationSpeed;
                this.portfolioMesh.rotation.x += this.rotationSpeed * 0.5;
            }
            
            // Animate position indicators
            if (this.positionIndicators) {
                this.positionIndicators.children.forEach((child, index) => {
                    const originalPos = child.userData.originalPosition;
                    const offset = Math.sin(this.time * 2 + index) * 0.5;
                    child.position.y = originalPos.y + offset;
                    child.rotation.y += 0.01;
                });
            }
            
            // Rotate particle system
            if (this.particleSystem) {
                this.particleSystem.rotation.y += 0.001;
            }
            
            // Update TWEEN animations
            if (typeof TWEEN !== 'undefined') {
                TWEEN.update();
            }
            
            this.renderer.render(this.scene, this.camera);
        };
        
        animate();
    }
    
    stopAnimation() {
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
            this.animationId = null;
        }
    }
    
    // Public API methods
    setPerformanceData(data) {
        this.updateVisualization({ performance: data });
    }
    
    setRiskData(data) {
        this.updateVisualization({ riskMetrics: data });
    }
    
    setPositions(positions) {
        this.updateVisualization({ positions: positions });
    }
    
    toggleWireframe() {
        if (this.portfolioMesh) {
            this.portfolioMesh.material.wireframe = !this.portfolioMesh.material.wireframe;
        }
    }
    
    resetCamera() {
        this.camera.position.set(0, 20, 40);
        this.camera.lookAt(0, 0, 0);
        this.controls.reset();
    }
    
    dispose() {
        this.stopAnimation();
        
        // Dispose of geometries and materials
        this.scene.traverse((child) => {
            if (child.geometry) {
                child.geometry.dispose();
            }
            if (child.material) {
                if (Array.isArray(child.material)) {
                    child.material.forEach(material => material.dispose());
                } else {
                    child.material.dispose();
                }
            }
        });
        
        this.renderer.dispose();
        
        const container = document.getElementById(this.containerId);
        if (container && this.renderer.domElement) {
            container.removeChild(this.renderer.domElement);
        }
    }
}

// Export for global use
window.Portfolio3DVisualization = Portfolio3DVisualization;

console.log('🎮 3D Visualization module loaded');
