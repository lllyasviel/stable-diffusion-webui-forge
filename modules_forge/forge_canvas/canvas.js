// ------------------------------------------------------------
// Utility Classes
// ------------------------------------------------------------
class GradioTextAreaBind {
    constructor(elementId, className) {
        this.target = document.querySelector(`#${elementId}.${className} textarea`);
        this.syncLock = false;
        this.previousValue = '';

        // In case the target isn't found, bail out early to avoid errors
        if (!this.target) {
            console.warn(`GradioTextAreaBind: Target textarea not found for #${elementId}.${className}`);
            return;
        }

        this.observer = new MutationObserver(() => {
            if (this.target.value !== this.previousValue) {
                this.previousValue = this.target.value;
                if (!this.syncLock && this._callback) {
                    this.syncLock = true;
                    this._callback(this.target.value);
                    this.syncLock = false;
                }
            }
        });

        // Observe changes
        this.observer.observe(this.target, {
            characterData: true,
            subtree: true,
            childList: true,
            attributes: true
        });
    }

    setValue(newValue) {
        if (!this.target) return;
        if (this.syncLock) return;

        this.syncLock = true;
        this.target.value = newValue;
        this.previousValue = newValue;

        const inputEvent = new Event('input', { bubbles: true });
        Object.defineProperty(inputEvent, 'target', { value: this.target });
        this.target.dispatchEvent(inputEvent);

        this.syncLock = false;
    }

    listen(callback) {
        this._callback = callback;
    }
}

// Add this class before ForgeCanvas class definition
class UndoManager {
    constructor(maxStates = 20) {
        this.states = [];
        this.currentIndex = -1;
        this.maxStates = maxStates;
    }

    pushState(state) {
        // Remove future states if we're in the middle of history
        if (this.currentIndex < this.states.length - 1) {
            this.states = this.states.slice(0, this.currentIndex + 1);
        }

        // Remove oldest state if at capacity
        if (this.states.length >= this.maxStates) {
            this.states.shift();
            this.currentIndex--;
        }

        this.states.push(state);
        this.currentIndex++;
    }

    canUndo() {
        return this.currentIndex > 0;
    }

    canRedo() {
        return this.currentIndex < this.states.length - 1;
    }

    undo() {
        if (!this.canUndo()) return null;
        this.currentIndex--;
        return this.states[this.currentIndex];
    }

    redo() {
        if (!this.canRedo()) return null;
        this.currentIndex++;
        return this.states[this.currentIndex];
    }

    clear() {
        this.states = [];
        this.currentIndex = -1;
    }

    getCurrentState() {
        return this.states[this.currentIndex];
    }
}

// ------------------------------------------------------------
// Main Canvas Class: ForgeCanvas
// ------------------------------------------------------------
class ForgeCanvas {
    constructor(
        uuid,
        noUpload = false,
        noScribbles = false,
        mask = false,
        initialHeight = 512,
        scribbleColor = '#000000',
        scribbleColorFixed = false,
        scribbleWidth = 4,
        scribbleWidthFixed = false,
        scribbleAlpha = 100,
        scribbleAlphaFixed = false,
        scribbleSoftness = 0,
        scribbleSoftnessFixed = false
    ) {
        this.gradioConfig = typeof gradio_config !== 'undefined' ? gradio_config : null;
        this.uuid = uuid;
        this.noUpload = noUpload;
        this.noScribbles = noScribbles;
        this.mask = mask;
        this.initialHeight = initialHeight;
        this.img = null;
        this.imgX = 0;
        this.imgY = 0;
        this.originalWidth = 0;
        this.originalHeight = 0;
        this.imgScale = 1;
        this.dragging = false;
        this.draggedJustNow = false;
        this.resizing = false;
        this.drawing = false;
        this.scribbleColor = scribbleColor;
        this.scribbleWidth = scribbleWidth;
        this.scribbleAlpha = scribbleAlpha;
        this.scribbleSoftness = scribbleSoftness;
        this.scribbleColorFixed = scribbleColorFixed;
        this.scribbleWidthFixed = scribbleWidthFixed;
        this.scribbleAlphaFixed = scribbleAlphaFixed;
        this.scribbleSoftnessFixed = scribbleSoftnessFixed;
        this.history = [];
        this.historyIndex = -1;
        this.maximized = false;
        this.originalState = {};
        this.contrastPattern = null;
        this.pointerInsideContainer = false;
        this.tempCanvas = document.createElement('canvas');
        this.tempDrawPoints = [];
        this.tempDrawBackground = null;
        this.backgroundGradioBind = new GradioTextAreaBind(this.uuid, 'logical_image_background');
        this.foregroundGradioBind = new GradioTextAreaBind(this.uuid, 'logical_image_foreground');
        this.contrastPatternCanvas = null;
        this.currentMode = 'normal';
        this.currentTool = 'brush';
        this.eraseChanged = false;
        this.toolbarDragging = false;
        this.toolbarOffset = { x: 0, y: 0 };
        this.undoManager = new UndoManager(20); // Limit to 20 states

        // *** CHANGES ***
        // For performance: keep track of drawing context
        this.drawingCtx = null;

        // requestAnimationFrame loop controls
        this.isDrawingLoopActive = false;
        this.drawPending = false;

        // We store all brush or eraser commands since last frame
        this.brushStrokes = []; // { x0, y0, x1, y1, type }

        // Debounce for dataURL updates
        this.uploadDebounceTimer = null;
        this.uploadDebounceDelay = 300; // ms

        this.start(); // Initialize all logic
    }

    /**
     * High-level initialization function.
     */
    start() {
        // 1. Cache and store DOM references
        this.cacheDOMElements();

        // 2. Initialize UI states
        this.initUI();

        // 3. Bind event handlers
        this.bindToolbarEvents();
        this.bindCanvasEvents();
        this.bindDragDropEvents();
        this.bindGlobalEvents();

        // 4. Set up watchers
        this.observeContainerResize();

        // 5. Final touches
        this.updateUndoRedoButtons();
        this.backgroundGradioBind.listen(base64Data => this.uploadBase64(base64Data));
        this.foregroundGradioBind.listen(base64Data => this.uploadBase64DrawingCanvas(base64Data));

        // Prevent default scroll on the drawing canvas
        if (this.drawingCanvas) {
            this.drawingCanvas.addEventListener('wheel', e => e.preventDefault(), { passive: false });
            this.drawingCanvas.setAttribute('tabindex', '0');
        }

        // *** CHANGES ***
        // Kick off an animation loop for drawing changes
        this.startDrawingLoop();
    }

    // ------------------------------------------------------------
    // 1) DOM CACHING & INITIAL UI SETUP
    // ------------------------------------------------------------
    cacheDOMElements() {
        const ids = [
            'imageContainer', 'image', 'resizeLine', 'container', 'toolbar', 'uploadButton',
            'resetButton', 'centerButton', 'removeButton', 'undoButton', 'redoButton',
            'drawingCanvas', 'maxButton', 'minButton', 'scribbleIndicator', 'uploadHint',
            'scribbleColor', 'scribbleColorBlock', 'scribbleWidth', 'widthLabel',
            'scribbleWidthBlock', 'scribbleAlpha', 'alphaLabel', 'scribbleAlphaBlock',
            'scribbleSoftness', 'softnessLabel', 'scribbleSoftnessBlock', 'eraserButton'
        ];

        this.elems = {};
        ids.forEach(id => {
            const el = document.getElementById(`${id}_${this.uuid}`);
            this.elems[id] = el || null;
        });
    }

    initUI() {
        const {
            scribbleColor, scribbleWidth, scribbleAlpha, scribbleSoftness,
            scribbleIndicator, container, drawingCanvas, uploadButton, uploadHint
        } = this.elems;

        // Initialize scribble controls
        if (scribbleColor) scribbleColor.value = this.scribbleColor;
        if (scribbleWidth) scribbleWidth.value = this.scribbleWidth;
        if (scribbleAlpha) scribbleAlpha.value = this.scribbleAlpha;
        if (scribbleSoftness) scribbleSoftness.value = this.scribbleSoftness;

        // Indicator size
        if (scribbleIndicator) {
            const scribbleIndicatorSize = this.scribbleWidth * 20;
            scribbleIndicator.style.width = `${scribbleIndicatorSize}px`;
            scribbleIndicator.style.height = `${scribbleIndicatorSize}px`;
        }

        // Container height
        if (container) container.style.height = `${this.initialHeight}px`;

        // Initialize drawing canvas
        if (drawingCanvas && this.elems.imageContainer) {
            drawingCanvas.width = this.elems.imageContainer.clientWidth;
            drawingCanvas.height = this.elems.imageContainer.clientHeight;
            this.drawingCanvas = drawingCanvas;
            // *** CHANGES ***
            // Grab and store a single 2D context
            this.drawingCtx = drawingCanvas.getContext('2d');
        }

        // Hide scribble-related elements if noScribbles is true
        if (this.noScribbles) {
            [
                'resetButton', 'undoButton', 'redoButton', 'scribbleColor', 'scribbleColorBlock',
                'scribbleWidthBlock', 'scribbleAlphaBlock', 'scribbleSoftnessBlock',
                'scribbleIndicator', 'drawingCanvas'
            ].forEach(id => {
                if (this.elems[id]) this.elems[id].style.display = 'none';
            });
        }

        // Hide upload button & hint if noUpload is true
        if (this.noUpload && uploadButton) {
            uploadButton.style.display = 'none';
            if (uploadHint) uploadHint.style.display = 'none';
        }

        // Mask mode
        if (this.mask) {
            this.configureMaskMode();
        }

        // Hide/fix scribble controls if flagged as fixed
        if (this.scribbleColorFixed && this.elems.scribbleColorBlock) {
            this.elems.scribbleColorBlock.style.display = 'none';
        }
        if (this.scribbleWidthFixed && this.elems.scribbleWidthBlock) {
            this.elems.scribbleWidthBlock.style.display = 'none';
        }
        if (this.scribbleAlphaFixed && this.elems.scribbleAlphaBlock) {
            this.elems.scribbleAlphaBlock.style.display = 'none';
        }
        if (this.scribbleSoftnessFixed && this.elems.scribbleSoftnessBlock) {
            this.elems.scribbleSoftnessBlock.style.display = 'none';
        }

        // Enhanced tooltips
        this.initTooltips();
    }

    configureMaskMode() {
        const { scribbleColorBlock, scribbleAlphaBlock, scribbleSoftnessBlock, drawingCanvas } = this.elems;

        // Hide color/alpha/softness controls
        [scribbleColorBlock, scribbleAlphaBlock, scribbleSoftnessBlock].forEach(el => {
            if (el) el.style.display = 'none';
        });

        // Create the contrast pattern
        if (drawingCanvas) {
            const patternCanvas = document.createElement('canvas');
            patternCanvas.width = 20;
            patternCanvas.height = 20;
            const patternContext = patternCanvas.getContext('2d');
            patternContext.fillStyle = '#ffffff';
            patternContext.fillRect(0, 0, 10, 10);
            patternContext.fillRect(10, 10, 10, 10);
            patternContext.fillStyle = '#000000';
            patternContext.fillRect(10, 0, 10, 10);
            patternContext.fillRect(0, 10, 10, 10);

            this.contrastPatternCanvas = patternCanvas;
            this.contrastPattern = this.drawingCtx.createPattern(patternCanvas, 'repeat');
            drawingCanvas.style.opacity = '0.5';
            this.currentMode = 'inpainting';
        }
    }

    initTooltips() {
        const tooltips = {
            uploadButton: 'Upload Image (or drag & drop)',
            resetButton: 'Reset Canvas (R)',
            undoButton: 'Undo (Ctrl+Z)',
            redoButton: 'Redo (Ctrl+Y)',
            eraserButton: 'Eraser Tool (E)',
            scribbleWidth: 'Brush Size ([ and ])',
        };

        Object.entries(tooltips).forEach(([id, tooltip]) => {
            const elem = this.elems[id];
            if (elem) elem.title = tooltip;
        });
    }

    // ------------------------------------------------------------
    // 2) EVENT BINDING
    // ------------------------------------------------------------
    bindToolbarEvents() {
        const {
            uploadButton, resetButton, centerButton, removeButton,
            undoButton, redoButton, maxButton, minButton
        } = this.elems;

        // File upload input
        const imageInput = document.getElementById(`imageInput_${this.uuid}`);
        if (imageInput) {
            imageInput.addEventListener('change', e => this.handleFileUpload(e.target.files[0]));
        }

        if (uploadButton && imageInput && !this.noUpload) {
            uploadButton.addEventListener('click', () => imageInput.click());
        }
        if (resetButton) {
            resetButton.addEventListener('click', () => this.resetImage());
        }
        if (centerButton) {
            centerButton.addEventListener('click', () => {
                this.adjustInitialPositionAndScale();
                this.drawImage();
            });
        }
        if (removeButton) {
            removeButton.addEventListener('click', () => this.removeImage());
        }
        if (undoButton) {
            undoButton.addEventListener('click', () => this.undo());
        }
        if (redoButton) {
            redoButton.addEventListener('click', () => this.redo());
        }
        if (maxButton) {
            maxButton.addEventListener('click', () => this.maximize());
        }
        if (minButton) {
            minButton.addEventListener('click', () => this.minimize());
        }

        // Draggable toolbar
        const toolbarHandle = document.getElementById(`toolbarHandle_${this.uuid}`);
        if (toolbarHandle && this.elems.toolbar) {
            toolbarHandle.addEventListener('mousedown', e => {
                this.toolbarDragging = true;
                const toolbarRect = this.elems.toolbar.getBoundingClientRect();
                this.toolbarOffset = {
                    x: e.clientX - toolbarRect.left,
                    y: e.clientY - toolbarRect.top
                };
                e.preventDefault();
            });

            document.addEventListener('mousemove', e => {
                if (!this.toolbarDragging || !this.elems.toolbar || !this.elems.imageContainer) return;
                const containerRect = this.elems.imageContainer.getBoundingClientRect();
                const toolbarRect = this.elems.toolbar.getBoundingClientRect();

                let newX = e.clientX - containerRect.left - this.toolbarOffset.x;
                let newY = e.clientY - containerRect.top - this.toolbarOffset.y;

                // Keep toolbar within container bounds
                newX = Math.max(0, Math.min(newX, containerRect.width - toolbarRect.width));
                newY = Math.max(0, Math.min(newY, containerRect.height - toolbarRect.height));

                this.elems.toolbar.style.left = `${newX}px`;
                this.elems.toolbar.style.top = `${newY}px`;
            });

            document.addEventListener('mouseup', () => {
                this.toolbarDragging = false;
            });
        }
    }

    bindCanvasEvents() {
        const {
            scribbleColor, scribbleIndicator, scribbleWidth, scribbleAlpha,
            scribbleSoftness, eraserButton, drawingCanvas, imageContainer, image
        } = this.elems;

        // Scribble color
        if (scribbleColor) {
            scribbleColor.addEventListener('input', () => {
                this.scribbleColor = scribbleColor.value;
                if (scribbleIndicator) {
                    scribbleIndicator.style.borderColor = this.scribbleColor;
                }
            });
        }

        // Scribble width
        if (scribbleWidth) {
            scribbleWidth.addEventListener('input', () => {
                this.scribbleWidth = parseInt(scribbleWidth.value, 10);
                const newSize = this.scribbleWidth * 20;
                if (scribbleIndicator) {
                    scribbleIndicator.style.width = `${newSize}px`;
                    scribbleIndicator.style.height = `${newSize}px`;
                }
            });
        }

        // Scribble alpha
        if (scribbleAlpha) {
            scribbleAlpha.addEventListener('input', () => {
                this.scribbleAlpha = parseInt(scribbleAlpha.value, 10);
            });
        }

        // Scribble softness
        if (scribbleSoftness) {
            scribbleSoftness.addEventListener('input', () => {
                this.scribbleSoftness = parseInt(scribbleSoftness.value, 10);
            });
        }

        // Eraser button
        if (eraserButton) {
            eraserButton.addEventListener('click', () => {
                this.currentTool = this.currentTool === 'eraser' ? 'brush' : 'eraser';
                if (this.mask && this.currentTool === 'brush' && this.drawingCanvas) {
                    this.drawingCtx.globalCompositeOperation = 'source-over';
                    this.drawingCtx.strokeStyle = this.contrastPattern;
                }
                eraserButton.classList.toggle('active');
            });
        }

        // *** CHANGES ***
        // We'll collect pointerdown and pointermove, but actual drawing
        // will happen via requestAnimationFrame in 'startDrawingLoop()'

        // Canvas pointer events
        if (drawingCanvas) {
            drawingCanvas.addEventListener('pointerdown', e => {
                if (!this.img || e.button !== 0 || this.noScribbles) return;

                this.drawing = true;
                drawingCanvas.style.cursor = 'crosshair';
                if (this.elems.scribbleIndicator) {
                    this.elems.scribbleIndicator.style.display = 'none';
                }
                this.tempDrawPoints = [];
                this.saveState(); // save once on start
                this.handlePointerMoveCanvas(e); // record first position
            });

            drawingCanvas.addEventListener('pointermove', e => {
                this.handlePointerMoveCanvas(e);
            });

            drawingCanvas.addEventListener('pointerup', () => {
                if (this.drawing) {
                    this.drawing = false;
                    this.lastErasePoint = null;
                    drawingCanvas.style.cursor = '';
                    if (this.eraseChanged) {
                        this.saveState();
                        this.eraseChanged = false;
                    }
                }
            });

            drawingCanvas.addEventListener('pointerleave', () => {
                this.drawing = false;
                this.lastErasePoint = null;
            });
        }

        // Image dragging inside container
        if (imageContainer) {
            imageContainer.addEventListener('pointerdown', e => this.onPointerDownImageContainer(e));
            imageContainer.addEventListener('pointermove', e => this.onPointerMoveImageContainer(e));
            imageContainer.addEventListener('pointerup', e => this.onPointerUpImageContainer(e));
            imageContainer.addEventListener('pointerleave', e => this.onPointerLeaveImageContainer(e));
            imageContainer.addEventListener('wheel', e => this.onWheelImageContainer(e), { passive: false });
            imageContainer.addEventListener('contextmenu', e => {
                e.preventDefault();
                this.draggedJustNow = false;
            });
            imageContainer.addEventListener('pointerover', () => {
                if (this.elems.toolbar) this.elems.toolbar.style.opacity = '1';
                if (!this.img && !this.noUpload && imageContainer) {
                    imageContainer.style.cursor = 'pointer';
                }
            });
            imageContainer.addEventListener('pointerout', () => {
                if (this.elems.toolbar) this.elems.toolbar.style.opacity = '0';
                if (image) image.style.cursor = '';
                if (drawingCanvas) drawingCanvas.style.cursor = '';
                if (imageContainer) imageContainer.style.cursor = '';
                if (scribbleIndicator) scribbleIndicator.style.display = 'none';
            });
        }

        // Resize line
        if (this.elems.resizeLine) {
            this.elems.resizeLine.addEventListener('pointerdown', e => {
                this.resizing = true;
                e.preventDefault();
                e.stopPropagation();
            });
        }
        document.addEventListener('pointermove', e => {
            if (this.resizing) {
                this.resizeContainer(e);
                e.preventDefault();
                e.stopPropagation();
            }
        });
        document.addEventListener('pointerup', () => {
            this.resizing = false;
        });
        document.addEventListener('pointerleave', () => {
            this.resizing = false;
        });
    }

    bindDragDropEvents() {
        const { imageContainer, image, drawingCanvas } = this.elems;
        if (!imageContainer) return;

        ['dragenter', 'dragover'].forEach(eventType => {
            imageContainer.addEventListener(eventType, e => e.preventDefault(), false);
        });

        // Visual feedback on drag
        imageContainer.addEventListener('dragenter', () => {
            if (image) image.style.cursor = 'copy';
            if (drawingCanvas) drawingCanvas.style.cursor = 'copy';
        });
        imageContainer.addEventListener('dragleave', () => {
            if (image) image.style.cursor = '';
            if (drawingCanvas) drawingCanvas.style.cursor = '';
        });

        // File drop
        imageContainer.addEventListener('drop', e => {
            e.preventDefault();
            const { dataTransfer } = e;
            const { files } = dataTransfer;
            if (files.length > 0) {
                this.handleFileUpload(files[0]);
            }
        });
    }

    bindGlobalEvents() {
        // Keyboard shortcuts
        document.addEventListener('keydown', e => {
            if (!this.pointerInsideContainer) return;
            if (e.key === 'b') this.setTool('brush');
            if (e.key === 'e') this.setTool('eraser');
            if (e.key === '[') this.adjustBrushSize(-1);
            if (e.key === ']') this.adjustBrushSize(1);
        });

        // Global undo/redo
        document.addEventListener('keydown', e => {
            if (e.ctrlKey && e.key === 'z') {
                e.preventDefault();
                this.undo();
            } else if (e.ctrlKey && e.key === 'y') {
                e.preventDefault();
                this.redo();
            }
        });

        // Pasting images
        document.addEventListener('paste', e => {
            if (this.pointerInsideContainer) {
                e.preventDefault();
                e.stopPropagation();
                this.handlePaste(e);
            }
        });

        // Track pointer inside container
        if (this.elems.imageContainer) {
            this.elems.imageContainer.addEventListener('pointerenter', () => {
                this.pointerInsideContainer = true;
            });
            this.elems.imageContainer.addEventListener('pointerleave', () => {
                this.pointerInsideContainer = false;
            });
        }
    }

    observeContainerResize() {
        if (!this.elems.container) return;
        const resizeObserver = new ResizeObserver(() => {
            this.adjustInitialPositionAndScale();
            this.drawImage();
        });
        resizeObserver.observe(this.elems.container);
    }

    // ------------------------------------------------------------
    // requestAnimationFrame-based drawing loop
    // ------------------------------------------------------------
    // *** CHANGES ***
    startDrawingLoop() {
        if (this.isDrawingLoopActive) return;
        this.isDrawingLoopActive = true;

        const drawFrame = () => {
            if (!this.drawPending || this.brushStrokes.length === 0) {
                // No new draws, skip
                this.drawPending = false;
            } else {
                // Perform the actual line drawing or erasing
                for (const stroke of this.brushStrokes) {
                    if (stroke.type === 'eraser') {
                        this.drawEraserLine(stroke.x0, stroke.y0, stroke.x1, stroke.y1);
                        this.eraseChanged = true;
                    } else {
                        this.drawBrushLine(stroke.x0, stroke.y0, stroke.x1, stroke.y1);
                    }
                }
                this.brushStrokes = []; // Clear
            }

            requestAnimationFrame(drawFrame);
        };

        requestAnimationFrame(drawFrame);
    }

    handlePointerMoveCanvas(e) {
        if (!this.drawing || !this.img || this.noScribbles) return;
        const rect = this.drawingCanvas.getBoundingClientRect();
        const x = (e.clientX - rect.left) / this.imgScale;
        const y = (e.clientY - rect.top) / this.imgScale;

        // We'll connect the last point to this point
        if (this.tempDrawPoints.length) {
            const [x0, y0] = this.tempDrawPoints[this.tempDrawPoints.length - 1];
            // Instead of immediate draw, queue it
            this.brushStrokes.push({
                x0, y0, x1: x, y1: y,
                type: this.currentTool === 'eraser' ? 'eraser' : 'brush'
            });
            this.drawPending = true;
        }
        this.tempDrawPoints.push([x, y]);
    }

    // Single line approach for brush
    // *** CHANGES ***
    drawBrushLine(x0, y0, x1, y1) {
        const ctx = this.drawingCtx;
        ctx.save();

        // For mask mode
        if (this.mask) {
            ctx.globalCompositeOperation = 'source-over';
            ctx.strokeStyle = this.contrastPattern || this.contrastPatternCanvas;
            ctx.lineWidth = (this.scribbleWidth / this.imgScale) * 20;
            ctx.lineCap = 'round';
            ctx.lineJoin = 'round';
            ctx.beginPath();
            ctx.moveTo(x0, y0);
            ctx.lineTo(x1, y1);
            ctx.stroke();
            ctx.restore();
            return;
        }

        // Use shadow for softness
        ctx.globalCompositeOperation = 'source-over';
        ctx.strokeStyle = this.scribbleColor;
        ctx.globalAlpha = this.scribbleAlpha / 100;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        ctx.lineWidth = (this.scribbleWidth / this.imgScale) * 20;

        // *** CHANGES ***
        // Use shadowBlur to simulate softness (instead of multi-pass)
        ctx.shadowColor = this.scribbleColor;
        ctx.shadowBlur = this.scribbleSoftness; // can be tuned

        ctx.beginPath();
        ctx.moveTo(x0, y0);
        ctx.lineTo(x1, y1);
        ctx.stroke();

        ctx.restore();
    }

    // Single line approach for eraser
    // *** CHANGES ***
    drawEraserLine(x0, y0, x1, y1) {
        const ctx = this.drawingCtx;
        ctx.save();
        ctx.globalCompositeOperation = 'destination-out';
        ctx.strokeStyle = 'rgba(0,0,0,1)';
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        ctx.lineWidth = (this.scribbleWidth / this.imgScale) * 20;

        ctx.beginPath();
        ctx.moveTo(x0, y0);
        ctx.lineTo(x1, y1);
        ctx.stroke();

        ctx.restore();
    }

    // ------------------------------------------------------------
    // 3) POINTER & MOUSE EVENT HANDLERS (Image Container)
    // ------------------------------------------------------------
    onPointerDownImageContainer(e) {
        const { imageContainer, image } = this.elems;
        if (!imageContainer || !this.img) {
            // If no image is loaded, possibly trigger upload
            if (!this.noUpload && e.button === 0) {
                const imageInput = document.getElementById(`imageInput_${this.uuid}`);
                if (imageInput) imageInput.click();
            }
            return;
        }

        const rect = imageContainer.getBoundingClientRect();
        const offsetX = e.clientX - rect.left;
        const offsetY = e.clientY - rect.top;

        // Right-click dragging
        if (e.button === 2 && this.isInsideImage(offsetX, offsetY)) {
            this.dragging = true;
            this.offsetX = offsetX - this.imgX;
            this.offsetY = offsetY - this.imgY;
            if (image) image.style.cursor = 'grabbing';
            if (this.drawingCanvas) this.drawingCanvas.style.cursor = 'grabbing';
            if (this.elems.scribbleIndicator) this.elems.scribbleIndicator.style.display = 'none';
        }
    }

    onPointerMoveImageContainer(e) {
        if (this.dragging) {
            const { imageContainer } = this.elems;
            if (!imageContainer) return;
            const rect = imageContainer.getBoundingClientRect();
            const mouseX = e.clientX - rect.left;
            const mouseY = e.clientY - rect.top;

            this.imgX = mouseX - this.offsetX;
            this.imgY = mouseY - this.offsetY;
            this.drawImage();
            this.draggedJustNow = true;
        }
    }

    onPointerUpImageContainer() {
        if (this.dragging) {
            this.handleDragEnd();
        }
    }

    onPointerLeaveImageContainer() {
        if (this.dragging) {
            this.handleDragEnd();
        }
    }

    onWheelImageContainer(e) {
        if (e.ctrlKey) {
            // Adjust brush size with Ctrl+wheel
            const brushChange = e.deltaY * -0.01;
            this.scribbleWidth = Math.max(1, this.scribbleWidth + brushChange);
            if (this.elems.scribbleWidth) this.elems.scribbleWidth.value = this.scribbleWidth;
            if (this.elems.scribbleIndicator) {
                const newSize = this.scribbleWidth * 20;
                this.elems.scribbleIndicator.style.width = `${newSize}px`;
                this.elems.scribbleIndicator.style.height = `${newSize}px`;
            }
            return;
        }

        if (!this.img) return;
        e.preventDefault();

        const { imageContainer } = this.elems;
        if (!imageContainer) return;
        const rect = imageContainer.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const mouseY = e.clientY - rect.top;
        const previousScale = this.imgScale;
        const zoomFactor = e.deltaY * -0.001;

        this.imgScale += zoomFactor;
        this.imgScale = Math.max(0.1, this.imgScale);
        const scaleRatio = this.imgScale / previousScale;

        this.imgX = mouseX - (mouseX - this.imgX) * scaleRatio;
        this.imgY = mouseY - (mouseY - this.imgY) * scaleRatio;

        this.drawImage();
    }

    resizeContainer(e) {
        const { container } = this.elems;
        if (!container) return;
        const containerRect = container.getBoundingClientRect();
        const newHeight = e.clientY - containerRect.top;
        container.style.height = `${newHeight}px`;
    }

    handleDragEnd() {
        this.dragging = false;
        if (this.elems.image) this.elems.image.style.cursor = 'grab';
        if (this.drawingCanvas) this.drawingCanvas.style.cursor = 'grab';
    }

    // ------------------------------------------------------------
    // 4) IMAGE / FILE / CLIPBOARD HANDLING
    // ------------------------------------------------------------
    handleFileUpload(file) {
        if (!file || this.noUpload) return;

        this.clearHistory();
        const reader = new FileReader();
        reader.onload = evt => this.uploadBase64(evt.target.result);
        reader.onerror = err => {
            console.error('FileReader error:', err);
        };

        try {
            reader.readAsDataURL(file);
        } catch (error) {
            console.error('Failed to read file:', error);
        }
    }

    handlePaste(e) {
        const { items } = e.clipboardData;
        for (let i = 0; i < items.length; i++) {
            const item = items[i];
            if (item.type.indexOf('image') !== -1) {
                const file = item.getAsFile();
                this.handleFileUpload(file);
                break;
            }
        }
    }

    uploadBase64(base64Data) {
        if (this.gradioConfig && !this.gradioConfig.version?.startsWith('4')) return;
        if (!this.gradioConfig) return;

        const img = this.tempImage || new Image();
        img.onload = () => {
            this.img = base64Data;
            this.originalWidth = img.width;
            this.originalHeight = img.height;

            const {drawingCanvas} = this.elems;
            if (drawingCanvas && (drawingCanvas.width !== img.width || drawingCanvas.height !== img.height)) {
                drawingCanvas.width = img.width;
                drawingCanvas.height = img.height;
            }

            this.adjustInitialPositionAndScale();
            this.drawImage();
            this.onImageUpload();
            this.saveState();

            const imageInput = document.getElementById(`imageInput_${this.uuid}`);
            if (imageInput) imageInput.value = null;
            if (this.elems.uploadHint) this.elems.uploadHint.style.display = 'none';
        };

        if (base64Data) {
            img.src = base64Data;
        } else {
            this.img = null;
            const {drawingCanvas} = this.elems;
            if (drawingCanvas) {
                drawingCanvas.width = 1;
                drawingCanvas.height = 1;
            }
            this.adjustInitialPositionAndScale();
            this.drawImage();
            this.onImageUpload();
            this.saveState();
        }
    }

    uploadBase64DrawingCanvas(base64Data) {
        const img = this.tempImage || new Image();
        img.onload = () => {
            const {drawingCanvas} = this.elems;
            if (!drawingCanvas) return;
            this.drawingCtx.clearRect(0, 0, drawingCanvas.width, drawingCanvas.height);
            this.drawingCtx.drawImage(img, 0, 0);
            this.saveState();
        };

        if (base64Data) {
            img.src = base64Data;
        } else {
            const {drawingCanvas} = this.elems;
            if (!drawingCanvas) return;
            this.drawingCtx.clearRect(0, 0, drawingCanvas.width, drawingCanvas.height);
            this.saveState();
        }
    }

    // ------------------------------------------------------------
    // 5) DRAWING & ERASING (No multi-pass softness)
    // ------------------------------------------------------------

    // In this version, everything is handled by the single-line approach above.

    // ------------------------------------------------------------
    // 6) CANVAS / IMAGE OPERATIONS & HISTORY
    // ------------------------------------------------------------
    drawImage() {
        const { image, drawingCanvas } = this.elems;
        if (!image || !drawingCanvas) return;

        if (this.img) {
            const scaledWidth = this.originalWidth * this.imgScale;
            const scaledHeight = this.originalHeight * this.imgScale;

            image.src = this.img;
            Object.assign(image.style, {
                width: `${scaledWidth}px`,
                height: `${scaledHeight}px`,
                left: `${this.imgX}px`,
                top: `${this.imgY}px`,
                display: 'block'
            });

            Object.assign(drawingCanvas.style, {
                width: `${scaledWidth}px`,
                height: `${scaledHeight}px`,
                left: `${this.imgX}px`,
                top: `${this.imgY}px`
            });
        } else {
            image.src = '';
            image.style.display = 'none';
        }
    }

    adjustInitialPositionAndScale() {
        const { imageContainer } = this.elems;
        if (!imageContainer || !this.originalWidth || !this.originalHeight) return;

        const containerWidth = imageContainer.clientWidth - 20;
        const containerHeight = imageContainer.clientHeight - 20;

        const scaleX = containerWidth / this.originalWidth;
        const scaleY = containerHeight / this.originalHeight;
        this.imgScale = Math.min(scaleX, scaleY);

        const scaledWidth = this.originalWidth * this.imgScale;
        const scaledHeight = this.originalHeight * this.imgScale;

        this.imgX = (imageContainer.clientWidth - scaledWidth) / 2;
        this.imgY = (imageContainer.clientHeight - scaledHeight) / 2;
    }

    resetImage() {
        const { drawingCanvas } = this.elems;
        if (!drawingCanvas) return;
        this.drawingCtx.clearRect(0, 0, drawingCanvas.width, drawingCanvas.height);

        this.adjustInitialPositionAndScale();
        this.drawImage();
        this.saveState();
    }

    removeImage() {
        this.img = null;
        const { image, drawingCanvas, uploadHint } = this.elems;
        if (drawingCanvas) {
            this.drawingCtx.clearRect(0, 0, drawingCanvas.width, drawingCanvas.height);
        }
        if (image) {
            image.src = '';
            image.style.width = '0';
            image.style.height = '0';
        }

        this.saveState();
        if (!this.noUpload && uploadHint) {
            uploadHint.style.display = 'block';
        }

        this.onImageUpload();
        this.clearHistory();
    }

    isInsideImage(x, y) {
        const scaledWidth = this.originalWidth * this.imgScale;
        const scaledHeight = this.originalHeight * this.imgScale;
        return (
            x > this.imgX &&
            x < this.imgX + scaledWidth &&
            y > this.imgY &&
            y < this.imgY + scaledHeight
        );
    }

    saveState() {
        const {drawingCanvas} = this.elems;
        if (!drawingCanvas) return;

        const state = {
            timestamp: Date.now(),
            imageData: this.drawingCtx.getImageData(0, 0, drawingCanvas.width, drawingCanvas.height)
        };

        this.undoManager.pushState(state);
        this.updateUndoRedoButtons();

        // *** CHANGES ***
        // Debounce dataURL generation
        if (this.uploadDebounceTimer) {
            clearTimeout(this.uploadDebounceTimer);
        }
        this.uploadDebounceTimer = setTimeout(() => {
            this.onDrawingCanvasUpload();
        }, this.uploadDebounceDelay);
    }

    undo() {
        const state = this.undoManager.undo();
        if (state) {
            this.applyState(state);
            this.updateUndoRedoButtons();
            // Force immediate dataURL
            this.onDrawingCanvasUpload(true);
        }
    }

    redo() {
        const state = this.undoManager.redo();
        if (state) {
            this.applyState(state);
            this.updateUndoRedoButtons();
            // Force immediate dataURL
            this.onDrawingCanvasUpload(true);
        }
    }

    updateUndoRedoButtons() {
        const { undoButton, redoButton } = this.elems;

        if (undoButton) {
            undoButton.disabled = !this.undoManager.canUndo();
            undoButton.style.opacity = undoButton.disabled ? '0.5' : '1';
        }

        if (redoButton) {
            redoButton.disabled = !this.undoManager.canRedo();
            redoButton.style.opacity = redoButton.disabled ? '0.5' : '1';
        }
    }

    applyState(state) {
        const {drawingCanvas} = this.elems;
        if (!drawingCanvas) return;
        this.drawingCtx.putImageData(state.imageData, 0, 0);
    }

    clearHistory() {
        this.undoManager.clear();
        this.updateUndoRedoButtons();
    }

    onImageUpload() {
        if (!this.img) {
            this.backgroundGradioBind.setValue('');
            return;
        }

        const { image } = this.elems;
        if (!image) return;

        const { tempCanvas } = this;
        const ctx = tempCanvas.getContext('2d');
        tempCanvas.width = this.originalWidth;
        tempCanvas.height = this.originalHeight;
        ctx.drawImage(image, 0, 0, this.originalWidth, this.originalHeight);

        const base64Data = tempCanvas.toDataURL('image/png');
        this.backgroundGradioBind.setValue(base64Data);
    }

    onDrawingCanvasUpload(forceImmediate = false) {
        if (!this.img) {
            this.foregroundGradioBind.setValue('');
            return;
        }
        const {drawingCanvas} = this.elems;
        if (!drawingCanvas) return;

        // Optionally skip the debounce if forceImmediate
        if (!forceImmediate) {
            if (this.uploadDebounceTimer) {
                clearTimeout(this.uploadDebounceTimer);
            }
            this.uploadDebounceTimer = setTimeout(() => {
                const base64Data = drawingCanvas.toDataURL('image/png');
                this.foregroundGradioBind.setValue(base64Data);
            }, this.uploadDebounceDelay);
        } else {
            const base64Data = drawingCanvas.toDataURL('image/png');
            this.foregroundGradioBind.setValue(base64Data);
        }
    }

    // ------------------------------------------------------------
    // 7) UI MAXIMIZE / MINIMIZE & TOOL SELECTION
    // ------------------------------------------------------------
    maximize() {
        if (this.maximized) return;

        const { container, maxButton, minButton } = this.elems;
        if (!container || !maxButton || !minButton) return;

        this.originalState = {
            width: container.style.width,
            height: container.style.height,
            top: container.style.top,
            left: container.style.left,
            position: container.style.position,
            zIndex: container.style.zIndex
        };

        Object.assign(container.style, {
            width: '100vw',
            height: '100vh',
            top: '0',
            left: '0',
            position: 'fixed',
            zIndex: '1000'
        });

        maxButton.style.display = 'none';
        minButton.style.display = 'inline-block';
        this.maximized = true;
    }

    minimize() {
        if (!this.maximized) return;

        const { container, maxButton, minButton } = this.elems;
        if (!container || !maxButton || !minButton) return;

        Object.assign(container.style, {
            width: this.originalState.width,
            height: this.originalState.height,
            top: this.originalState.top,
            left: this.originalState.left,
            position: this.originalState.position,
            zIndex: this.originalState.zIndex
        });

        maxButton.style.display = 'inline-block';
        minButton.style.display = 'none';
        this.maximized = false;
    }

    setTool(tool) {
        if (tool === this.currentTool) return;
        this.currentTool = tool;

        const { eraserButton } = this.elems;
        if (eraserButton) {
            eraserButton.classList.toggle('active', tool === 'eraser');
        }
    }

    adjustBrushSize(delta) {
        const newWidth = Math.max(1, Math.min(100, this.scribbleWidth + delta));
        this.scribbleWidth = newWidth;

        const input = this.elems.scribbleWidth;
        if (input) input.value = newWidth;

        const indicator = this.elems.scribbleIndicator;
        if (indicator) {
            const size = newWidth * 20;
            indicator.style.width = `${size}px`;
            indicator.style.height = `${size}px`;
        }
    }
}

// Constants
const True = true;
const False = false;