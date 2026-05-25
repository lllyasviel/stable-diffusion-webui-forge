(function() {
    const MOBILE_QUERY = "(max-width: 760px), (pointer: coarse) and (max-width: 900px)";
    const GENERATION_TABS = ["txt2img", "img2img"];
    const SCREENS = ["compose", "params", "assets", "output"];
    const ASSET_STAGES = ["groups", "preview", "items"];
    const SCREEN_LABELS = {
        compose: "Prompt",
        params: "Tune",
        assets: "Assets",
        output: "Output"
    };
    const ASSET_STAGE_LABELS = {
        groups: "Groups",
        preview: "Preview",
        items: "Items"
    };

    const state = {
        active: false,
        tab: "txt2img",
        screen: "compose",
        assetStage: "groups",
        drawerOpen: false,
        selectedGroup: "",
        allowCardCommit: false,
        cart: new Set()
    };

    let mobileMedia = null;
    let shell = null;
    let cardListenerRoot = null;
    let groupListenerRoot = null;
    let updateTimer = null;
    let lastTaggedCount = 0;

    function app() {
        return gradioApp();
    }

    function qsa(selector, root) {
        return Array.from((root || app()).querySelectorAll(selector));
    }

    function byId(id) {
        return app().getElementById(id);
    }

    function currentTopTabContent() {
        return app().querySelector('#tabs > .tabitem[id^=tab_]:not([style*="display: none"])');
    }

    function activeGenerationTab() {
        const content = currentTopTabContent();

        if (content && content.id === "tab_img2img") {
            return "img2img";
        }

        if (content && content.id === "tab_txt2img") {
            return "txt2img";
        }

        return state.tab;
    }

    function currentTabRoot() {
        return byId("tab_" + state.tab) || app();
    }

    function button(label, action, variant) {
        const el = document.createElement("button");
        el.type = "button";
        el.textContent = label;
        el.dataset.mobileAction = action;
        if (variant) {
            el.classList.add(variant);
        }
        return el;
    }

    function setButtonState(el, selected) {
        el.classList.toggle("selected", selected);
        el.setAttribute("aria-pressed", selected ? "true" : "false");
    }

    function ensureShell() {
        if (shell && shell.isConnected) {
            return shell;
        }

        shell = document.createElement("div");
        shell.id = "mobile_screen_shell";
        shell.setAttribute("aria-hidden", "true");
        shell.innerHTML = [
            '<div class="mobile-screen-drawer" hidden></div>',
            '<div class="mobile-screen-context">',
            '    <div class="mobile-screen-title"></div>',
            '    <div class="mobile-screen-actions"></div>',
            '</div>',
            '<button class="mobile-screen-primary" type="button"></button>',
            '<nav class="mobile-screen-nav" aria-label="Mobile screens"></nav>'
        ].join("");

        const nav = shell.querySelector(".mobile-screen-nav");
        nav.appendChild(button("Prompt", "screen:compose"));
        nav.appendChild(button("Tune", "screen:params"));
        nav.appendChild(button("Assets", "screen:assets"));
        nav.appendChild(button("Output", "screen:output"));
        nav.appendChild(button("Mode", "mode"));

        shell.addEventListener("click", onShellClick);
        document.body.appendChild(shell);
        return shell;
    }

    function mobileEnabled() {
        return mobileMedia && mobileMedia.matches;
    }

    function setActive(active) {
        state.active = active;
        document.body.classList.toggle("mobile-screen-mode", active);

        if (shell) {
            shell.setAttribute("aria-hidden", active ? "false" : "true");
        }

        if (!active) {
            document.body.removeAttribute("data-mobile-screen");
            document.body.removeAttribute("data-mobile-asset-stage");
            clearCart();
            return;
        }

        state.tab = activeGenerationTab();
        ensureScreenTab();
        syncMobileMode();
    }

    function scheduleSync() {
        clearTimeout(updateTimer);
        updateTimer = setTimeout(syncMobileMode, 80);
    }

    function tagPanels() {
        let taggedCount = 0;

        GENERATION_TABS.forEach(function(tab) {
            const topRow = byId(tab + "_toprow");
            const promptContainer = byId(tab + "_prompt_container");
            const settings = byId(tab + "_settings");
            const results = byId(tab + "_results");
            const imageMode = tab === "img2img" ? byId("mode_img2img") : null;
            const inpaintControls = tab === "img2img" ? byId("inpaint_controls") : null;

            if (topRow) {
                topRow.dataset.mobilePanel = "compose";
                taggedCount += 1;
            }

            if (promptContainer) {
                promptContainer.dataset.mobilePanel = "compose";
                taggedCount += 1;
            }

            if (settings) {
                settings.dataset.mobilePanelRoot = "settings";

                Array.from(settings.children).forEach(function(child) {
                    if (child === promptContainer || child.querySelector("#" + tab + "_prompt_container")) {
                        child.dataset.mobilePanel = "compose";
                    } else if (child === imageMode || (imageMode && child.contains(imageMode))) {
                        child.dataset.mobilePanel = "params image";
                    } else {
                        child.dataset.mobilePanel = "params";
                    }
                    taggedCount += 1;
                });
            }

            if (imageMode) {
                imageMode.dataset.mobilePanel = "params image";
                taggedCount += 1;
            }

            if (inpaintControls) {
                inpaintControls.dataset.mobilePanel = "params image";
                taggedCount += 1;
            }

            if (results) {
                results.dataset.mobilePanel = "output";
                taggedCount += 1;
            }

            qsa("#" + tab + "_extra_tabs > .tabitem.extra-page").forEach(function(page) {
                page.dataset.mobilePanel = "assets";
                taggedCount += 1;
            });
        });

        lastTaggedCount = taggedCount;
    }

    function setPanelVisibility() {
        qsa("[data-mobile-panel]").forEach(function(el) {
            const panels = (el.dataset.mobilePanel || "").split(/\s+/);
            let active = panels.indexOf(state.screen) !== -1;

            if (active && state.screen === "assets" && el.classList.contains("extra-page")) {
                active = el.style.display !== "none";
            }

            el.classList.toggle("mobile-panel-active", active);
        });
    }

    function topTabButtons() {
        return qsa("#tabs > .tab-nav > button");
    }

    function clickTopTab(tab) {
        const buttonById = byId("tab_" + tab + "-button");

        if (buttonById) {
            buttonById.click();
            return;
        }

        const tabs = topTabButtons();
        const index = tab === "img2img" ? 1 : 0;

        if (tabs[index]) {
            tabs[index].click();
        }
    }

    function extraTabButtons(tab) {
        return qsa("#" + tab + "_extra_tabs > .tab-nav > button");
    }

    function extraSelectedIndex(tab) {
        const buttons = extraTabButtons(tab);

        for (let i = 0; i < buttons.length; i += 1) {
            if (buttons[i].classList.contains("selected") || buttons[i].getAttribute("aria-selected") === "true") {
                return i;
            }
        }

        return 0;
    }

    function ensureScreenTab() {
        const topTab = activeGenerationTab();
        if (topTab !== state.tab) {
            state.tab = topTab;
        }

        if (state.screen === "assets") {
            const buttons = extraTabButtons(state.tab);
            const selected = extraSelectedIndex(state.tab);

            if (buttons.length > 1 && selected === 0) {
                buttons[1].click();
            }
            return;
        }

        const generation = extraTabButtons(state.tab)[0];
        if (generation && extraSelectedIndex(state.tab) !== 0) {
            generation.click();
        }
    }

    function setScreen(screen) {
        if (SCREENS.indexOf(screen) === -1) {
            return;
        }

        state.screen = screen;
        if (screen === "assets" && ASSET_STAGES.indexOf(state.assetStage) === -1) {
            state.assetStage = "groups";
        }
        ensureScreenTab();
        syncMobileMode();
    }

    function setAssetStage(stage) {
        if (ASSET_STAGES.indexOf(stage) === -1) {
            return;
        }

        state.screen = "assets";
        state.assetStage = stage;
        ensureScreenTab();
        syncMobileMode();
    }

    function goRelative(delta) {
        if (state.screen === "assets") {
            const stageIndex = ASSET_STAGES.indexOf(state.assetStage);
            const nextStage = ASSET_STAGES[stageIndex + delta];

            if (nextStage) {
                setAssetStage(nextStage);
                return;
            }
        }

        const index = SCREENS.indexOf(state.screen);
        const next = SCREENS[index + delta];

        if (next) {
            setScreen(next);
        }
    }

    function clearCart() {
        state.cart.forEach(function(card) {
            card.classList.remove("mobile-cart-selected");
            card.removeAttribute("aria-pressed");
        });
        state.cart.clear();
    }

    function cleanCart() {
        Array.from(state.cart).forEach(function(card) {
            if (!card.isConnected || card.classList.contains("hidden")) {
                card.classList.remove("mobile-cart-selected");
                state.cart.delete(card);
            }
        });
    }

    function toggleCard(card) {
        if (state.cart.has(card)) {
            state.cart.delete(card);
            card.classList.remove("mobile-cart-selected");
            card.setAttribute("aria-pressed", "false");
        } else {
            state.cart.add(card);
            card.classList.add("mobile-cart-selected");
            card.setAttribute("aria-pressed", "true");
        }

        syncMobileMode();
    }

    function commitCart() {
        cleanCart();

        if (!state.cart.size) {
            return;
        }

        const cards = Array.from(state.cart);
        state.allowCardCommit = true;
        cards.forEach(function(card) {
            if (card.isConnected) {
                card.click();
            }
        });
        state.allowCardCommit = false;
        clearCart();
        setScreen("compose");
    }

    function onCardClick(event) {
        if (!state.active || state.screen !== "assets" || state.assetStage !== "items" || state.allowCardCommit) {
            return;
        }

        const card = event.target.closest(".extra-network-pane .card");
        if (!card) {
            return;
        }

        event.preventDefault();
        event.stopPropagation();
        event.stopImmediatePropagation();
        toggleCard(card);
    }

    function attachCardListener() {
        const root = app();
        if (cardListenerRoot === root) {
            return;
        }

        if (cardListenerRoot) {
            cardListenerRoot.removeEventListener("click", onCardClick, true);
        }

        cardListenerRoot = root;
        cardListenerRoot.addEventListener("click", onCardClick, true);
    }

    function groupLabelFromTarget(target) {
        const label = target.querySelector(".tree-list-item-label");

        if (label) {
            return label.textContent.trim();
        }

        return target.textContent.trim();
    }

    function onGroupClick(event) {
        if (!state.active || state.screen !== "assets") {
            return;
        }

        const target = event.target.closest(".extra-network-tree .tree-list-content, .extra-network-dirs button, #" + state.tab + "_extra_tabs > .tab-nav > button");
        if (!target) {
            return;
        }

        state.selectedGroup = groupLabelFromTarget(target);
        if (state.assetStage === "groups") {
            setTimeout(function() {
                setAssetStage("preview");
            }, 0);
        } else {
            scheduleSync();
        }
    }

    function attachGroupListener() {
        const root = app();
        if (groupListenerRoot === root) {
            return;
        }

        if (groupListenerRoot) {
            groupListenerRoot.removeEventListener("click", onGroupClick, true);
        }

        groupListenerRoot = root;
        groupListenerRoot.addEventListener("click", onGroupClick, true);
    }

    function visibleCard(card) {
        return !card.classList.contains("hidden");
    }

    function updatePreviewCards() {
        qsa(".extra-network-pane .card").forEach(function(card) {
            card.classList.remove("mobile-preview-card");
        });

        if (state.screen !== "assets" || state.assetStage !== "preview") {
            return;
        }

        const cards = qsa("#" + state.tab + "_extra_tabs .extra-network-pane .card").filter(visibleCard);
        cards.slice(0, 8).forEach(function(card) {
            card.classList.add("mobile-preview-card");
        });
    }

    function clickScoped(selector) {
        const root = currentTabRoot();
        const target = root.querySelector(selector);

        if (target) {
            target.click();
            return true;
        }

        return false;
    }

    function inputScoped(selector) {
        return currentTabRoot().querySelector(selector);
    }

    function updateInputIfAvailable(input) {
        if (input && typeof updateInput === "function") {
            updateInput(input);
        }
    }

    function addPromptGroup() {
        const textarea = inputScoped("#" + state.tab + "_prompt textarea");

        if (!textarea) {
            return;
        }

        const start = textarea.selectionStart || textarea.value.length;
        const end = textarea.selectionEnd || textarea.value.length;
        const selected = textarea.value.slice(start, end);
        const insert = selected ? "(" + selected + ")" : (textarea.value ? ", ()" : "()");
        const cursor = selected ? start + insert.length : start + insert.length - 1;

        textarea.value = textarea.value.slice(0, start) + insert + textarea.value.slice(end);
        textarea.focus();
        textarea.setSelectionRange(cursor, cursor);
        updateInputIfAvailable(textarea);
    }

    function clickGenerateLike() {
        const interrupt = byId(state.tab + "_interrupt");
        const interrupting = byId(state.tab + "_interrupting");
        const generate = byId(state.tab + "_generate");

        if (interrupting && interrupting.style.display === "block") {
            interrupting.click();
            return;
        }

        if (interrupt && interrupt.style.display === "block") {
            interrupt.click();
            return;
        }

        if (generate) {
            generate.click();
        }
    }

    function clickSkip() {
        const skip = byId(state.tab + "_skip");

        if (skip) {
            skip.click();
        }
    }

    function cycleMode() {
        const next = state.tab === "txt2img" ? "img2img" : "txt2img";
        state.tab = next;
        clickTopTab(next);
        setTimeout(function() {
            state.tab = next;
            ensureScreenTab();
            scheduleSync();
        }, 0);
    }

    function toggleDrawer() {
        state.drawerOpen = !state.drawerOpen;
        syncShell();
    }

    function cycleExtraType() {
        const buttons = extraTabButtons(state.tab);

        if (buttons.length <= 2) {
            return;
        }

        const selected = extraSelectedIndex(state.tab);
        const next = selected + 1 >= buttons.length ? 1 : selected + 1;
        buttons[next].click();
        state.selectedGroup = buttons[next].textContent.trim();
        setAssetStage("groups");
    }

    function refreshAssets() {
        const selected = extraSelectedIndex(state.tab);
        const buttons = extraTabButtons(state.tab);
        const selectedButton = buttons[selected];

        if (selectedButton && selectedButton.id) {
            const pageId = selectedButton.id.replace(/-button$/, "");
            const refresh = byId(pageId + "_extra_refresh");
            if (refresh) {
                refresh.click();
            }
        }
    }

    function focusAssetSearch() {
        const selected = extraSelectedIndex(state.tab);
        const buttons = extraTabButtons(state.tab);
        const selectedButton = buttons[selected];

        if (selectedButton && selectedButton.id) {
            const pageId = selectedButton.id.replace(/-button$/, "");
            const search = byId(pageId + "_extra_search");
            if (search) {
                search.focus();
            }
        }
    }

    function onShellClick(event) {
        const action = event.target.dataset.mobileAction;

        if (!action) {
            return;
        }

        if (action.indexOf("screen:") === 0) {
            setScreen(action.slice("screen:".length));
            return;
        }

        if (action.indexOf("asset:") === 0) {
            setAssetStage(action.slice("asset:".length));
            return;
        }

        const actions = {
            back: function() {
                goRelative(-1);
            },
            next: function() {
                goRelative(1);
            },
            generate: clickGenerateLike,
            skip: clickSkip,
            mode: toggleDrawer,
            switchMode: cycleMode,
            paste: function() {
                clickScoped("#paste");
            },
            clear: function() {
                clickScoped("#" + state.tab + "_clear_prompt");
            },
            styles: function() {
                clickScoped("#" + state.tab + "_style_apply");
            },
            groupPrompt: addPromptGroup,
            swapSize: function() {
                clickScoped("#" + state.tab + "_res_switch_btn");
            },
            detectSize: function() {
                clickScoped("#img2img_detect_image_size_btn");
            },
            type: cycleExtraType,
            refresh: refreshAssets,
            focusSearch: focusAssetSearch,
            clearCart: function() {
                clearCart();
                syncMobileMode();
            },
            addCart: commitCart,
            restore: function() {
                clickScoped("#" + state.tab + "_restore_progress");
            },
            upscale: function() {
                clickScoped("#txt2img_upscale");
            },
            save: function() {
                clickScoped("#save_" + state.tab);
            },
            openFolder: function() {
                clickScoped("#" + state.tab + "_open_folder");
            }
        };

        if (actions[action]) {
            actions[action]();
        }
    }

    function addAction(container, label, action, variant) {
        container.appendChild(button(label, action, variant));
    }

    function renderActions(container) {
        container.innerHTML = "";

        addAction(container, "Back", "back");

        if (state.screen === "compose") {
            addAction(container, "Paste", "paste");
            addAction(container, "Clear", "clear");
            addAction(container, "Styles", "styles");
            addAction(container, "Group", "groupPrompt");
            addAction(container, "Next", "next", "primary");
            return;
        }

        if (state.screen === "params") {
            addAction(container, "Swap", "swapSize");
            if (state.tab === "img2img") {
                addAction(container, "Detect", "detectSize");
            }
            addAction(container, "Next", "next", "primary");
            return;
        }

        if (state.screen === "assets") {
            addAction(container, "Type", "type");
            addAction(container, "Groups", "asset:groups", state.assetStage === "groups" ? "primary" : "");
            addAction(container, "Preview", "asset:preview", state.assetStage === "preview" ? "primary" : "");
            addAction(container, "Items", "asset:items", state.assetStage === "items" ? "primary" : "");

            if (state.assetStage === "groups") {
                addAction(container, "Search", "focusSearch");
                addAction(container, "Refresh", "refresh");
            } else {
                addAction(container, "Clear", "clearCart");
            }
            return;
        }

        if (state.screen === "output") {
            addAction(container, "Restore", "restore");
            addAction(container, "Save", "save");
            addAction(container, "Folder", "openFolder");
            if (state.tab === "txt2img") {
                addAction(container, "Upscale", "upscale");
            }
        }
    }

    function updatePrimaryButton(primary) {
        const interrupt = byId(state.tab + "_interrupt");
        const interrupting = byId(state.tab + "_interrupting");
        const generating = (interrupt && interrupt.style.display === "block") || (interrupting && interrupting.style.display === "block");

        primary.dataset.mobileAction = "generate";
        primary.classList.toggle("generating", generating);

        if (state.screen === "assets" && state.assetStage === "items") {
            cleanCart();
            primary.textContent = state.cart.size ? "Add " + state.cart.size : (generating ? "Stop" : "Generate");
            primary.dataset.mobileAction = state.cart.size ? "addCart" : "generate";
            primary.disabled = false;
            return;
        }

        primary.disabled = false;
        primary.textContent = generating ? "Stop" : "Generate";
    }

    function renderDrawer(drawer) {
        drawer.hidden = !state.drawerOpen;
        drawer.innerHTML = "";

        topTabButtons().forEach(function(tabButton) {
            const item = button(tabButton.textContent.trim(), "drawer-tab", "");
            item.addEventListener("click", function() {
                tabButton.click();
                state.drawerOpen = false;
                scheduleSync();
            });
            drawer.appendChild(item);
        });
    }

    function syncShell() {
        const currentShell = ensureShell();
        const navButtons = qsa(".mobile-screen-nav button", currentShell);
        const actions = currentShell.querySelector(".mobile-screen-actions");
        const title = currentShell.querySelector(".mobile-screen-title");
        const primary = currentShell.querySelector(".mobile-screen-primary");
        const drawer = currentShell.querySelector(".mobile-screen-drawer");

        navButtons.forEach(function(navButton) {
            const action = navButton.dataset.mobileAction || "";
            setButtonState(navButton, action === "screen:" + state.screen || (action === "mode" && state.drawerOpen));
        });

        title.textContent = [
            state.tab === "txt2img" ? "Txt2img" : "Img2img",
            SCREEN_LABELS[state.screen],
            state.screen === "assets" ? ASSET_STAGE_LABELS[state.assetStage] : "",
            state.screen === "assets" && state.selectedGroup ? state.selectedGroup : ""
        ].filter(Boolean).join(" / ");

        renderActions(actions);
        updatePrimaryButton(primary);
        renderDrawer(drawer);
    }

    function syncMobileMode() {
        if (!state.active) {
            return;
        }

        const taggedBefore = lastTaggedCount;
        tagPanels();

        if (taggedBefore !== lastTaggedCount) {
            attachCardListener();
            attachGroupListener();
        }

        state.tab = activeGenerationTab();
        document.body.dataset.mobileScreen = state.screen;
        document.body.dataset.mobileAssetStage = state.assetStage;

        ensureScreenTab();
        setPanelVisibility();
        updatePreviewCards();
        syncShell();
    }

    function setup() {
        mobileMedia = window.matchMedia(MOBILE_QUERY);
        ensureShell();
        attachCardListener();
        attachGroupListener();

        const onChange = function() {
            setActive(mobileEnabled());
        };

        if (mobileMedia.addEventListener) {
            mobileMedia.addEventListener("change", onChange);
        } else {
            mobileMedia.addListener(onChange);
        }

        onChange();
    }

    onUiLoaded(setup);
    onAfterUiUpdate(function() {
        if (state.active) {
            scheduleSync();
        }
    });
})();
