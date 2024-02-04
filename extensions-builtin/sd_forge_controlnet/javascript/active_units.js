/**
 * Give a badge on ControlNet Accordion indicating total number of active
 * units.
 * Make active unit's tab name green.
 * Append control type to tab name.
 * Disable resize mode selection when A1111 img2img input is used.
 */
(function () {
    const cnetAllAccordions = new Set();
    onUiUpdate(() => {
        const ImgChangeType = {
            NO_CHANGE: 0,
            REMOVE: 1,
            ADD: 2,
            SRC_CHANGE: 3,
        };

        function imgChangeObserved(mutationsList) {
            // Iterate over all mutations that just occured
            for (let mutation of mutationsList) {
                // Check if the mutation is an addition or removal of a node
                if (mutation.type === 'childList') {
                    // Check if nodes were added
                    if (mutation.addedNodes.length > 0) {
                        for (const node of mutation.addedNodes) {
                            if (node.tagName === 'IMG') {
                                return ImgChangeType.ADD;
                            }
                        }
                    }

                    // Check if nodes were removed
                    if (mutation.removedNodes.length > 0) {
                        for (const node of mutation.removedNodes) {
                            if (node.tagName === 'IMG') {
                                return ImgChangeType.REMOVE;
                            }
                        }
                    }
                }
                // Check if the mutation is a change of an attribute
                else if (mutation.type === 'attributes') {
                    if (mutation.target.tagName === 'IMG' && mutation.attributeName === 'src') {
                        return ImgChangeType.SRC_CHANGE;
                    }
                }
            }
            return ImgChangeType.NO_CHANGE;
        }

        function childIndex(element) {
            // Get all child nodes of the parent
            let children = Array.from(element.parentNode.childNodes);

            // Filter out non-element nodes (like text nodes and comments)
            children = children.filter(child => child.nodeType === Node.ELEMENT_NODE);

            return children.indexOf(element);
        }

        function imageInputDisabledAlert() {
            alert('Inpaint control type must use a1111 input in img2img mode.');
        }

        class ControlNetUnitTab {
            constructor(tab, accordion) {
                this.tab = tab;
                this.tabOpen = false; // Whether the tab is open.
                this.accordion = accordion;
                this.isImg2Img = tab.querySelector('.cnet-mask-upload').id.includes('img2img');

                this.enabledAccordionCheckbox = tab.querySelector('.input-accordion-checkbox');
                this.enabledCheckbox = tab.querySelector('.cnet-unit-enabled input');
                this.inputImage = tab.querySelector('.cnet-input-image-group .cnet-image input[type="file"]');
                this.inputImageContainer = tab.querySelector('.cnet-input-image-group .cnet-image');
                this.generatedImageGroup = tab.querySelector('.cnet-generated-image-group');
                this.maskImageGroup = tab.querySelector('.cnet-mask-image-group');
                this.inputImageGroup = tab.querySelector('.cnet-input-image-group');
                this.controlTypeRadios = tab.querySelectorAll('.controlnet_control_type_filter_group input[type="radio"]');
                this.resizeModeRadios = tab.querySelectorAll('.controlnet_resize_mode_radio input[type="radio"]');
                this.runPreprocessorButton = tab.querySelector('.cnet-run-preprocessor');

                this.tabs = tab.parentNode;
                this.tabIndex = childIndex(tab);

                // By default the InputAccordion checkbox is linked with the state
                // of accordion's open/close state. To disable this link, we can
                // simulate click to check the checkbox and uncheck it.
                this.enabledAccordionCheckbox.click();
                this.enabledAccordionCheckbox.click();

                this.sync_enabled_checkbox();
                this.attachEnabledButtonListener();
                this.attachControlTypeRadioListener();
                this.attachImageUploadListener();
                this.attachImageStateChangeObserver();
                this.attachA1111SendInfoObserver();
                this.attachPresetDropdownObserver();
                this.attachAccordionStateObserver();
            }

            /**
             * Sync the states of enabledCheckbox and enabledAccordionCheckbox.
             */
            sync_enabled_checkbox() {
                this.enabledCheckbox.addEventListener("change", () => {
                    if (this.enabledAccordionCheckbox.checked != this.enabledCheckbox.checked) {
                        this.enabledAccordionCheckbox.click();
                    }
                });
                this.enabledAccordionCheckbox.addEventListener("change", () => {
                    if (this.enabledCheckbox.checked != this.enabledAccordionCheckbox.checked) {
                        this.enabledCheckbox.click();
                    }
                });
            }
            /**
             * Get the span that has text "Unit {X}".
             */
            getUnitHeaderTextElement() {
                return this.tab.querySelector(
                    `:nth-child(${this.tabIndex + 1}) span.svelte-s1r2yt`
                );
            }

            getActiveControlType() {
                for (let radio of this.controlTypeRadios) {
                    if (radio.checked) {
                        return radio.value;
                    }
                }
                return undefined;
            }

            updateActiveState() {
                const unitHeader = this.getUnitHeaderTextElement();
                if (!unitHeader) return;

                if (this.enabledCheckbox.checked) {
                    unitHeader.classList.add('cnet-unit-active');
                } else {
                    unitHeader.classList.remove('cnet-unit-active');
                }
            }

            updateActiveUnitCount() {
                function getActiveUnitCount(checkboxes) {
                    let activeUnitCount = 0;
                    for (const checkbox of checkboxes) {
                        if (checkbox.checked)
                            activeUnitCount++;
                    }
                    return activeUnitCount;
                }

                const checkboxes = this.accordion.querySelectorAll('.cnet-unit-enabled input');
                const span = this.accordion.querySelector('.label-wrap span');

                // Remove existing badge.
                if (span.childNodes.length !== 1) {
                    span.removeChild(span.lastChild);
                }
                // Add new badge if necessary.
                const activeUnitCount = getActiveUnitCount(checkboxes);
                if (activeUnitCount > 0) {
                    const div = document.createElement('div');
                    div.classList.add('cnet-badge');
                    div.classList.add('primary');
                    div.innerHTML = `${activeUnitCount} unit${activeUnitCount > 1 ? 's' : ''}`;
                    span.appendChild(div);
                }
            }

            /**
             * Add the active control type to tab displayed text.
             */
            updateActiveControlType() {
                const unitHeader = this.getUnitHeaderTextElement();
                if (!unitHeader) return;

                // Remove the control if exists
                const controlTypeSuffix = unitHeader.querySelector('.control-type-suffix');
                if (controlTypeSuffix) controlTypeSuffix.remove();

                // Add new suffix.
                const controlType = this.getActiveControlType();
                if (controlType === 'All') return;

                const span = document.createElement('span');
                span.innerHTML = `[${controlType}]`;
                span.classList.add('control-type-suffix');
                unitHeader.appendChild(span);
            }
            getInputImageSrc() {
                const img = this.inputImageGroup.querySelector('.cnet-image img');
                return img ? img.src : null;
            }
            getPreprocessorPreviewImageSrc() {
                const img = this.generatedImageGroup.querySelector('.cnet-image img');
                return img ? img.src : null;
            }
            getMaskImageSrc() {
                function isEmptyCanvas(canvas) {
                    if (!canvas) return true;
                    const ctx = canvas.getContext('2d');
                    // Get the image data
                    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
                    const data = imageData.data; // This is a Uint8ClampedArray
                    // Check each pixel
                    let isPureBlack = true;
                    for (let i = 0; i < data.length; i += 4) {
                        if (data[i] !== 0 || data[i + 1] !== 0 || data[i + 2] !== 0) { // Check RGB values
                            isPureBlack = false;
                            break;
                        }
                    }
                    return isPureBlack;
                }
                const maskImg = this.maskImageGroup.querySelector('.cnet-mask-image img');
                // Hand-drawn mask on mask upload.
                const handDrawnMaskCanvas = this.maskImageGroup.querySelector('.cnet-mask-image canvas[key="mask"]');
                // Hand-drawn mask on input image upload.
                const inputImageHandDrawnMaskCanvas = this.inputImageGroup.querySelector('.cnet-image canvas[key="mask"]');
                if (!isEmptyCanvas(handDrawnMaskCanvas)) {
                    return handDrawnMaskCanvas.toDataURL();
                } else if (maskImg) {
                    return maskImg.src;
                } else if (!isEmptyCanvas(inputImageHandDrawnMaskCanvas)) {
                    return inputImageHandDrawnMaskCanvas.toDataURL();
                } else {
                    return null;
                }
            }
            setThumbnail(imgSrc, maskSrc) {
                if (!imgSrc) return;
                const unitHeader = this.getUnitHeaderTextElement();
                if (!unitHeader) return;
                const img = document.createElement('img');
                img.src = imgSrc;
                img.classList.add('cnet-thumbnail');
                unitHeader.appendChild(img);

                if (maskSrc) {
                    const mask = document.createElement('img');
                    mask.src = maskSrc;
                    mask.classList.add('cnet-thumbnail');
                    unitHeader.appendChild(mask);
                }
            }
            removeThumbnail() {
                const unitHeader = this.getUnitHeaderTextElement();
                if (!unitHeader) return;
                const imgs = unitHeader.querySelectorAll('.cnet-thumbnail');
                for (const img of imgs) {
                    img.remove();
                }
            }
            /**
             * When the accordion is folded, display a thumbnail of input image
             * and mask on the accordion header.
             */
            updateInputImageThumbnail() {
                if (!opts.controlnet_input_thumbnail) return;
                if (this.tabOpen) {
                    this.removeThumbnail();
                } else {
                    this.setThumbnail(this.getInputImageSrc(), this.getMaskImageSrc());
                }
            }

            attachEnabledButtonListener() {
                this.enabledCheckbox.addEventListener('change', () => {
                    this.updateActiveState();
                    this.updateActiveUnitCount();
                });
            }

            attachControlTypeRadioListener() {
                for (const radio of this.controlTypeRadios) {
                    radio.addEventListener('change', () => {
                        this.updateActiveControlType();
                    });
                }
            }

            attachImageUploadListener() {
                // Automatically check `enable` checkbox when image is uploaded.
                this.inputImage.addEventListener('change', (event) => {
                    if (!event.target.files) return;
                    if (!this.enabledCheckbox.checked)
                        this.enabledCheckbox.click();
                });

                // Automatically check `enable` checkbox when JSON pose file is uploaded.
                this.tab.querySelector('.cnet-upload-pose input').addEventListener('change', (event) => {
                    if (!event.target.files) return;
                    if (!this.enabledCheckbox.checked)
                        this.enabledCheckbox.click();
                });
            }

            attachImageStateChangeObserver() {
                new MutationObserver((mutationsList) => {
                    const changeObserved = imgChangeObserved(mutationsList);

                    if (changeObserved === ImgChangeType.ADD) {
                        // enabling the run preprocessor button
                        this.runPreprocessorButton.removeAttribute("disabled");
                        this.runPreprocessorButton.title = 'Run preprocessor';
                    }

                    if (changeObserved === ImgChangeType.REMOVE) {
                        // disabling the run preprocessor button
                        this.runPreprocessorButton.setAttribute("disabled", true);
                        this.runPreprocessorButton.title = "No ControlNet input image available";
                    }
                }).observe(this.inputImageContainer, {
                    childList: true,
                    subtree: true,
                });
            }

            /**
             * Observe send PNG info buttons in A1111, as they can also directly
             * set states of ControlNetUnit.
             */
            attachA1111SendInfoObserver() {
                const pasteButtons = gradioApp().querySelectorAll('#paste');
                const pngButtons = gradioApp().querySelectorAll(
                    this.isImg2Img ?
                        '#img2img_tab, #inpaint_tab' :
                        '#txt2img_tab'
                );

                for (const button of [...pasteButtons, ...pngButtons]) {
                    button.addEventListener('click', () => {
                        // The paste/send img generation info feature goes
                        // though gradio, which is pretty slow. Ideally we should
                        // observe the event when gradio has done the job, but
                        // that is not an easy task.
                        // Here we just do a 2 second delay until the refresh.
                        setTimeout(() => {
                            this.updateActiveState();
                            this.updateActiveUnitCount();
                        }, 2000);
                    });
                }
            }

            attachPresetDropdownObserver() {
                const presetDropDown = this.tab.querySelector('.cnet-preset-dropdown');

                new MutationObserver((mutationsList) => {
                    for (const mutation of mutationsList) {
                        if (mutation.removedNodes.length > 0) {
                            setTimeout(() => {
                                this.updateActiveState();
                                this.updateActiveUnitCount();
                                this.updateActiveControlType();
                            }, 1000);
                            return;
                        }
                    }
                }).observe(presetDropDown, {
                    childList: true,
                    subtree: true,
                });
            }
            /**
             * Observer that triggers when the ControlNetUnit's accordion(tab) closes.
             */
            attachAccordionStateObserver() {
                new MutationObserver((mutationsList) => {
                    for(const mutation of mutationsList) {
                        if (mutation.type === 'attributes' && mutation.attributeName === 'class') {
                            const newState = mutation.target.classList.contains('open');
                            if (this.tabOpen != newState) {
                                this.tabOpen = newState;
                                if (newState) {
                                    this.onAccordionOpen();
                                } else {
                                    this.onAccordionClose();
                                }
                            }
                        }
                    }
                }).observe(this.tab.querySelector('.label-wrap'), { attributes: true, attributeFilter: ['class'] });
            }

            onAccordionOpen() {
                this.updateInputImageThumbnail();
            }

            onAccordionClose() {
                this.updateInputImageThumbnail();
            }
        }

        gradioApp().querySelectorAll('#controlnet').forEach(accordion => {
            if (cnetAllAccordions.has(accordion)) return;
            const tabs = [...accordion.querySelectorAll('.input-accordion')]
                .map(tab => new ControlNetUnitTab(tab, accordion));

            // On open of main extension accordion, if no unit is enabled,
            // open unit 0 for edit.
            const labelWrap = accordion.querySelector('.label-wrap');
            const observerAccordionOpen = new MutationObserver(function (mutations) {
                for (const mutation of mutations) {
                    if (mutation.target.classList.contains('open') &&
                        tabs.every(tab => !tab.enabledCheckbox.checked &&
                                          !tab.tab.querySelector('.label-wrap').classList.contains('open'))
                    ) {
                        tabs[0].tab.querySelector('.label-wrap').click();
                    }
                }
            });
            observerAccordionOpen.observe(labelWrap, { attributes: true, attributeFilter: ['class'] });

            cnetAllAccordions.add(accordion);
        });
    });
})();