/**
 * Aspect Ratio Previewer for Stable Diffusion Web UI (img2img tabs)
 *
 * Enhancements:
 * - Persistence Toggle: Keeps the preview visible.
 * - Info Display: Shows WxH and Aspect Ratio.
 * - Customizable Style: Via CSS variables.
 * - Debounced Updates: Improves performance on slider input.
 * - Window Resize Handling: Updates preview on resize.
 * - Structured Code: Encapsulated in an object.
 * - Robustness: More checks, cleaner listener management.
 */

const AspectRatioPreviewer = {
    // --- Configuration ---
    config: {
        debounceDelay: 150, // ms delay for slider/resize updates
        hideTimeoutDuration: 2000, // ms before hiding non-persistent preview
        selectors: {
            gradioApp: () => typeof gradioApp === 'function' ? gradioApp() : document.body,
            img2imgTab: "#tab_img2img",
            widthSliderParent: "#img2img_width",
            heightSliderParent: "#img2img_height",
            // Target image containers for each img2img mode (add more if needed)
            targetImageContainers: [
                '#img2img_image div[class*=image-container] img',      // img2img (Forge/A1111)
                '#img2img_image div[data-testid=image] img',           // img2img (Gradio 4 Image)
                '#img2img_sketch div[class*=image-container] img',     // Sketch
                '#img2img_sketch div[data-testid=image] img',          // Sketch (Gradio 4 Image)
                '#img2maskimg div[class*=image-container] img',        // Inpaint Mask
                '#img2maskimg div[data-testid=image] img',             // Inpaint Mask (Gradio 4 Image)
                '#inpaint_sketch div[class*=image-container] img',     // Inpaint Sketch
                '#inpaint_sketch div[data-testid=image] img',          // Inpaint Sketch (Gradio 4 Image)
                '#img_inpaint_base div[data-testid=image] img',        // Inpaint Upload Base
                '#img_inpaint_mask div[data-testid=image] img',        // Inpaint Upload Mask
            ],
            // Gradio 3 number input parent structure
            g3NumberInputParent: (el) => el.parentElement?.parentElement?.parentElement,
        },
        ids: {
            previewBox: "arPreviewBox",
            infoBox: "arPreviewInfoBox",
            persistenceToggle: "arPreviewPersistenceToggle",
            persistenceLabel: "arPreviewPersistenceLabel"
        },
        cssVars: {
            borderColor: '--ar-preview-border-color',
            borderStyle: '--ar-preview-border-style',
            borderWidth: '--ar-preview-border-width',
            infoBgColor: '--ar-preview-info-bg-color',
            infoTextColor: '--ar-preview-info-text-color',
        }
    },

    // --- State ---
    currentWidth: 512,
    currentHeight: 512,
    targetElement: null,
    arPreviewRect: null,
    arInfoBox: null,
    persistenceToggle: null,
    isPersistent: false,
    isVisible: false,
    hideTimeoutHandle: null,
    debounceTimeoutHandle: null,
    initialized: false,
    observer: null, // For watching tab visibility changes

    // --- Initialization ---
    init: function() {
        if (this.initialized) return; // Prevent multiple initializations

        console.log("Initializing AspectRatioPreviewer...");

        this.createPreviewElements();
        this.setupTabObserver(); // Use observer for reliability

        // Add window resize listener
        window.addEventListener('resize', () => this.debounceUpdate());

        this.initialized = true;
        // Perform initial setup if img2img is already visible
        this.handleVisibilityChange();
    },

    // Use MutationObserver to detect when the img2img tab becomes visible/hidden
    setupTabObserver: function() {
        const rootApp = this.config.selectors.gradioApp();
        const img2imgTab = rootApp.querySelector(this.config.selectors.img2imgTab);

        if (!img2imgTab) {
            console.warn("AspectRatioPreviewer: Could not find img2img tab:", this.config.selectors.img2imgTab);
            return;
        }

        this.observer = new MutationObserver((mutationsList) => {
            for (const mutation of mutationsList) {
                if (mutation.type === 'attributes' && mutation.attributeName === 'style') {
                    this.handleVisibilityChange();
                }
            }
        });

        this.observer.observe(img2imgTab, { attributes: true });
    },

    // Handles changes in the img2img tab's visibility
    handleVisibilityChange: function() {
        const rootApp = this.config.selectors.gradioApp();
        const img2imgTab = rootApp.querySelector(this.config.selectors.img2imgTab);

        if (img2imgTab && img2imgTab.style.display === "block") {
            console.log("AspectRatioPreviewer: img2img tab visible, setting up inputs.");
            this.setupInputListeners();
            // Trigger initial calculation if needed
            this.updateDimensions(this.currentWidth, this.currentHeight, true);
        } else {
            console.log("AspectRatioPreviewer: img2img tab hidden.");
            this.hidePreview(true); // Force hide when tab is not visible
            // Note: Input listeners remain attached, but dimensionChange checks visibility.
        }
    },

    // Creates the preview div, info box, and persistence toggle
    createPreviewElements: function() {
        const rootApp = this.config.selectors.gradioApp();

        // Preview Rectangle
        if (!rootApp.querySelector(`#${this.config.ids.previewBox}`)) {
            this.arPreviewRect = document.createElement('div');
            this.arPreviewRect.id = this.config.ids.previewBox;
            // Apply default styles (can be overridden by CSS)
            this.arPreviewRect.style.position = 'absolute';
            this.arPreviewRect.style.boxSizing = 'border-box';
            this.arPreviewRect.style.pointerEvents = 'none';
            this.arPreviewRect.style.zIndex = '1000';
            this.arPreviewRect.style.borderWidth = `var(${this.config.cssVars.borderWidth}, 2px)`;
            this.arPreviewRect.style.borderStyle = `var(${this.config.cssVars.borderStyle}, solid)`;
            this.arPreviewRect.style.borderColor = `var(${this.config.cssVars.borderColor}, rgba(255, 0, 0, 0.75))`;
            this.arPreviewRect.style.display = 'none'; // Start hidden
            rootApp.appendChild(this.arPreviewRect);
        } else {
            this.arPreviewRect = rootApp.querySelector(`#${this.config.ids.previewBox}`);
        }

        // Info Box (inside preview rectangle)
        if (!this.arPreviewRect.querySelector(`#${this.config.ids.infoBox}`)) {
            this.arInfoBox = document.createElement('div');
            this.arInfoBox.id = this.config.ids.infoBox;
            // Apply default styles
            this.arInfoBox.style.position = 'absolute';
            this.arInfoBox.style.bottom = '2px';
            this.arInfoBox.style.left = '2px';
            this.arInfoBox.style.padding = '1px 4px';
            this.arInfoBox.style.fontSize = '10px';
            this.arInfoBox.style.borderRadius = '2px';
            this.arInfoBox.style.backgroundColor = `var(${this.config.cssVars.infoBgColor}, rgba(0, 0, 0, 0.6))`;
            this.arInfoBox.style.color = `var(${this.config.cssVars.infoTextColor}, white)`;
            this.arInfoBox.style.whiteSpace = 'nowrap';
            this.arPreviewRect.appendChild(this.arInfoBox);
        } else {
            this.arInfoBox = this.arPreviewRect.querySelector(`#${this.config.ids.infoBox}`);
        }

        // Persistence Toggle Checkbox & Label
        const sliderParent = rootApp.querySelector(this.config.selectors.widthSliderParent);
        if (sliderParent && !rootApp.querySelector(`#${this.config.ids.persistenceToggle}`)) {
            const container = document.createElement('div');
            container.style.display = 'flex';
            container.style.alignItems = 'center';
            container.style.marginTop = '5px'; // Add some spacing

            this.persistenceToggle = document.createElement('input');
            this.persistenceToggle.type = 'checkbox';
            this.persistenceToggle.id = this.config.ids.persistenceToggle;
            this.persistenceToggle.style.marginRight = '5px';
            this.persistenceToggle.addEventListener('change', (e) => {
                this.isPersistent = e.target.checked;
                if (!this.isPersistent && this.isVisible) {
                    // If toggled off while visible, start hide timer
                    this.startHideTimeout();
                } else if (this.isPersistent && this.isVisible) {
                     // If toggled on, clear any existing hide timer
                    clearTimeout(this.hideTimeoutHandle);
                    this.hideTimeoutHandle = null;
                }
                console.log("AspectRatioPreviewer: Persistence set to", this.isPersistent);
            });

            const label = document.createElement('label');
            label.htmlFor = this.config.ids.persistenceToggle;
            label.id = this.config.ids.persistenceLabel;
            label.textContent = 'Keep AR Preview Visible';
            label.style.fontSize = '12px';
            label.style.margin = '0';
            label.style.userSelect = 'none';
            label.style.cursor = 'pointer';


            container.appendChild(this.persistenceToggle);
            container.appendChild(label);

            // Insert after the width slider/input container
            sliderParent.parentNode.insertBefore(container, sliderParent.nextSibling);
        } else if (rootApp.querySelector(`#${this.config.ids.persistenceToggle}`)) {
             this.persistenceToggle = rootApp.querySelector(`#${this.config.ids.persistenceToggle}`);
             // Re-attach listener in case of UI rebuild
             if (!this.persistenceToggle.dataset.listenerAttached) {
                 this.persistenceToggle.addEventListener('change', (e) => { this.isPersistent = e.target.checked; /* ... rest of logic ... */ });
                 this.persistenceToggle.dataset.listenerAttached = 'true';
             }
        }
    },

    // Finds relevant input elements and attaches listeners if not already done
    setupInputListeners: function() {
        const rootApp = this.config.selectors.gradioApp();
        const inputs = rootApp.querySelectorAll('input[type="range"], input[type="number"]');

        inputs.forEach(input => {
            const parentId = input.parentElement?.id;
            const g3Parent = this.config.selectors.g3NumberInputParent(input);
            const g3ParentId = g3Parent?.id;

            const isWidth = (parentId === this.config.selectors.widthSliderParent.substring(1) && input.type === "range") ||
                           (g3ParentId === this.config.selectors.widthSliderParent.substring(1) && input.type === "number");
            const isHeight = (parentId === this.config.selectors.heightSliderParent.substring(1) && input.type === "range") ||
                            (g3ParentId === this.config.selectors.heightSliderParent.substring(1) && input.type === "number");

            if (isWidth || isHeight) {
                // Use a data attribute to prevent adding listeners multiple times
                if (!input.dataset.arListenerAttached) {
                    input.addEventListener('input', (e) => {
                        const value = parseFloat(e.target.value);
                        if (!isNaN(value)) {
                             if (isWidth) this.debounceUpdate(value, null);
                             if (isHeight) this.debounceUpdate(null, value);
                        }
                    });
                    input.dataset.arListenerAttached = 'true';
                    console.log("AspectRatioPreviewer: Attached listener to", input);
                }

                // Update initial values
                const initialValue = parseFloat(input.value);
                 if (!isNaN(initialValue)) {
                     if (isWidth) this.currentWidth = initialValue;
                     if (isHeight) this.currentHeight = initialValue;
                 }
            }
        });
         console.log(`AspectRatioPreviewer: Initial dimensions W: ${this.currentWidth}, H: ${this.currentHeight}`);
    },

    // --- Update Logic ---

    // Debounced wrapper for updateDimensions
    debounceUpdate: function(width = null, height = null) {
        clearTimeout(this.debounceTimeoutHandle);
        this.debounceTimeoutHandle = setTimeout(() => {
            this.updateDimensions(width, height);
        }, this.config.debounceDelay);
    },

    // Core update function, called after debounce
    updateDimensions: function(width = null, height = null, forceUpdate = false) {
        if (width !== null) this.currentWidth = width;
        if (height !== null) this.currentHeight = height;

        // Check if img2img tab is active
        const rootApp = this.config.selectors.gradioApp();
        const img2imgTab = rootApp.querySelector(this.config.selectors.img2imgTab);
        if (!img2imgTab || img2imgTab.style.display !== "block") {
            this.hidePreview(true); // Ensure hidden if tab not visible
            return;
        }

        // Find the currently active target image element
        this.targetElement = this.findTargetElement();

        if (this.targetElement) {
            // If the image source changed or dimensions are zero, might need re-calc after load
            if (this.targetElement.naturalWidth === 0 || this.targetElement.naturalHeight === 0) {
                 console.warn("AspectRatioPreviewer: Target image dimensions are zero. Waiting for load?");
                 // Optionally, attach a one-time load listener
                 if (!this.targetElement.dataset.loadListenerAttached) {
                     this.targetElement.addEventListener('load', () => this.calculateAndShowPreview(), { once: true });
                     this.targetElement.dataset.loadListenerAttached = 'true';
                 }
                 this.hidePreview(); // Hide until loaded
                 return;
            }
            // Remove load listener flag if already loaded
            delete this.targetElement.dataset.loadListenerAttached;

            this.calculateAndShowPreview();
        } else {
            // Hide preview if no valid target image found
            this.hidePreview();
        }
    },

    findTargetElement: function() {
        const rootApp = this.config.selectors.gradioApp();
        let currentTarget = null;
        // Try finding based on visible container first for better reliability
        for (const selector of this.config.selectors.targetImageContainers) {
            const element = rootApp.querySelector(selector);
            // Check if element exists and its container is visible
            if (element && element.offsetParent !== null) { // Check if it's actually visible
                currentTarget = element;
                break;
            }
        }

        // Fallback using tab index if needed (less reliable if UI structure changes)
        if (!currentTarget && typeof get_tab_index === 'function') {
            try {
                const tabIndex = get_tab_index('mode_img2img'); // Ensure 'mode_img2img' is correct ID
                const selectorsByIndex = [
                    '#img2img_image div[class*=image-container] img', // 0: img2img
                    '#img2img_sketch div[class*=image-container] img', // 1: Sketch
                    '#img2maskimg div[class*=image-container] img',    // 2: Inpaint Mask
                    '#inpaint_sketch div[class*=image-container] img', // 3: Inpaint sketch
                    '#img_inpaint_base div[data-testid=image] img',    // 4: Inpaint upload base
                ];
                 // Gradio 4 selectors as fallback
                 const selectorsByIndexG4 = [
                    '#img2img_image div[data-testid=image] img',
                    '#img2img_sketch div[data-testid=image] img',
                    '#img2maskimg div[data-testid=image] img',
                    '#inpaint_sketch div[data-testid=image] img',
                    '#img_inpaint_base div[data-testid=image] img',
                 ];

                if (tabIndex >= 0 && tabIndex < selectorsByIndex.length) {
                    currentTarget = rootApp.querySelector(selectorsByIndex[tabIndex]) || rootApp.querySelector(selectorsByIndexG4[tabIndex]);
                }
            } catch (e) {
                console.error("AspectRatioPreviewer: Error getting tab index.", e);
            }
        }

         if (!currentTarget) {
             // console.warn("AspectRatioPreviewer: Could not find visible target image element.");
         }
        return currentTarget;
    },

    // --- Calculation & Display ---

    calculateAndShowPreview: function() {
        if (!this.targetElement || !this.arPreviewRect || this.currentWidth <= 0 || this.currentHeight <= 0 || !this.targetElement.complete) {
            this.hidePreview();
            return;
        }

        const img = this.targetElement;
        const rect = this.arPreviewRect;
        const info = this.arInfoBox;

        try {
            const viewportOffset = img.getBoundingClientRect();
            // Use naturalWidth/Height for calculations, clientWidth/Height for container bounds
            const naturalW = img.naturalWidth;
            const naturalH = img.naturalHeight;

            if (naturalW === 0 || naturalH === 0 || img.clientWidth === 0 || img.clientHeight === 0) {
                console.warn("AspectRatioPreviewer: Image dimensions are zero, cannot calculate preview.");
                this.hidePreview();
                return;
            }

            // Calculate how the browser scales the image to fit its container
            const viewportScale = Math.min(img.clientWidth / naturalW, img.clientHeight / naturalH);

            // Actual displayed size of the image content within the img element
            const displayedW = naturalW * viewportScale;
            const displayedH = naturalH * viewportScale;

            // Calculate the offset of the displayed image content within the img element bounds (due to object-fit: contain)
            const offsetX = (img.clientWidth - displayedW) / 2;
            const offsetY = (img.clientHeight - displayedH) / 2;

            // Get the absolute position of the img element container
            const containerTop = viewportOffset.top + window.scrollY;
            const containerLeft = viewportOffset.left + window.scrollX;

            // Calculate the center of the *actual displayed image content*
            const imgContentCenterX = containerLeft + offsetX + (displayedW / 2);
            const imgContentCenterY = containerTop + offsetY + (displayedH / 2);

            // Calculate the scale factor for the AR preview to fit within the displayed image
            const arScale = Math.min(displayedW / this.currentWidth, displayedH / this.currentHeight);

            // Calculate the final scaled dimensions of the AR preview box
            const arScaledW = this.currentWidth * arScale;
            const arScaledH = this.currentHeight * arScale;

            // Calculate top-left position to center the AR preview box over the displayed image content
            const arRectTop = imgContentCenterY - (arScaledH / 2);
            const arRectLeft = imgContentCenterX - (arScaledW / 2);

            // Apply styles to the preview rectangle
            rect.style.top = `${arRectTop}px`;
            rect.style.left = `${arRectLeft}px`;
            rect.style.width = `${arScaledW}px`;
            rect.style.height = `${arScaledH}px`;

            // Update and show info box
            if (info) {
                const aspectRatio = (this.currentWidth / this.currentHeight).toFixed(2);
                info.textContent = `${this.currentWidth}x${this.currentHeight} (${aspectRatio})`;
                info.style.display = 'block';
            }

            this.showPreview();

        } catch (error) {
            console.error("AspectRatioPreviewer: Error calculating preview:", error);
            this.hidePreview();
        }
    },

    showPreview: function() {
        if (!this.arPreviewRect) return;
        this.arPreviewRect.style.display = 'block';
        this.isVisible = true;
        // Clear any previous hide timeout
        clearTimeout(this.hideTimeoutHandle);
        this.hideTimeoutHandle = null;
        // Start hide timeout only if not persistent
        if (!this.isPersistent) {
            this.startHideTimeout();
        }
    },

    hidePreview: function(force = false) {
        // Don't hide if persistent, unless forced (e.g., tab change)
        if (this.isPersistent && !force) {
            return;
        }
        if (this.arPreviewRect) {
            this.arPreviewRect.style.display = 'none';
        }
        this.isVisible = false;
        clearTimeout(this.hideTimeoutHandle); // Clear any pending hide
        this.hideTimeoutHandle = null;
    },

    startHideTimeout: function() {
        clearTimeout(this.hideTimeoutHandle); // Clear previous timer
        this.hideTimeoutHandle = setTimeout(() => {
            this.hidePreview();
        }, this.config.hideTimeoutDuration);
    },
};

// --- Global Initialization ---
// Use onAfterUiUpdate to initialize and handle UI refreshes
onAfterUiUpdate(function() {
    // Initialize on first appropriate UI update
    if (!AspectRatioPreviewer.initialized) {
         // Small delay to ensure Gradio elements are fully ready after update
         setTimeout(() => {
              try {
                   AspectRatioPreviewer.init();
              } catch (e) {
                   console.error("AspectRatioPreviewer: Error during initialization.", e);
              }
         }, 100); // 100ms delay, adjust if needed
    } else {
         // Re-run setup/checks if already initialized, in case UI elements were recreated
          try {
               // Re-find elements that might have been replaced
               AspectRatioPreviewer.createPreviewElements(); // Re-find/create preview, info, toggle
               AspectRatioPreviewer.handleVisibilityChange(); // Re-check visibility and inputs
          } catch (e) {
               console.error("AspectRatioPreviewer: Error during UI update handling.", e);
          }
    }
});

// --- Optional CSS Customization ---
/* Add this to your user.css or via an extension's CSS:
:root {
  --ar-preview-border-color: rgba(0, 255, 0, 0.8);
  --ar-preview-border-style: dashed;
  --ar-preview-border-width: 1px;
  --ar-preview-info-bg-color: rgba(0, 0, 0, 0.7);
  --ar-preview-info-text-color: #eee;
}

#arPreviewBox {
  // Add any other custom styles, like box-shadow
  // box-shadow: 0 0 5px 1px var(--ar-preview-border-color);
}
*/
