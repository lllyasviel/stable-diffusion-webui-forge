let promptTokenCountUpdateFunctions = {};

function update_txt2img_tokens(...args) {
    // Called from Gradio
    update_token_counter("txt2img_token_button");
    update_token_counter("txt2img_negative_token_button");
    if (args.length == 2) {
        return args[0];
    }
    return args;
}

function update_img2img_tokens(...args) {
    // Called from Gradio
    update_token_counter("img2img_token_button");
    update_token_counter("img2img_negative_token_button");
    if (args.length == 2) {
        return args[0];
    }
    return args;
}

function update_token_counter(button_id) {
    promptTokenCountUpdateFunctions[button_id]?.();
}


function recalculatePromptTokens(name) {
    promptTokenCountUpdateFunctions[name]?.();
}

function recalculate_prompts_txt2img() {
    // Called from Gradio
    recalculatePromptTokens('txt2img_prompt');
    recalculatePromptTokens('txt2img_neg_prompt');
    return Array.from(arguments);
}

function recalculate_prompts_img2img() {
    // Called from Gradio
    recalculatePromptTokens('img2img_prompt');
    recalculatePromptTokens('img2img_neg_prompt');
    return Array.from(arguments);
}

function setupSingleTokenCounting(id, id_counter, id_button) {
    var prompt = gradioApp().getElementById(id);
    var counter = gradioApp().getElementById(id_counter);
    var textarea = gradioApp().querySelector(`#${id} > label > textarea`);

    if (counter.parentElement === prompt.parentElement) {
        return;
    }

    prompt.parentElement.insertBefore(counter, prompt);
    prompt.parentElement.style.position = "relative";

    var func = onEdit(id, textarea, 800, function() {
        if (counter.classList.contains("token-counter-visible")) {
            gradioApp().getElementById(id_button)?.click();
        }
    });
    promptTokenCountUpdateFunctions[id] = func;
    promptTokenCountUpdateFunctions[id_button] = func;
}

function setupDualTokenCounting(id, id_t5_counter, id_clip_counter, id_button) {
    var prompt = gradioApp().getElementById(id);
    var t5_counter = gradioApp().getElementById(id_t5_counter);
    var clip_counter = gradioApp().getElementById(id_clip_counter);
    var textarea = gradioApp().querySelector(`#${id} > label > textarea`);

    if (t5_counter.parentElement === prompt.parentElement && clip_counter.parentElement === prompt.parentElement) {
        return;
    }

    prompt.parentElement.insertBefore(t5_counter, prompt);
    prompt.parentElement.insertBefore(clip_counter, prompt);
    prompt.parentElement.style.position = "relative";

    var func = onEdit(id, textarea, 800, function() {
        if (t5_counter.classList.contains("token-counter-visible") || clip_counter.classList.contains("token-counter-visible")) {
            gradioApp().getElementById(id_button)?.click();
        }
    });
    promptTokenCountUpdateFunctions[id] = func;
    promptTokenCountUpdateFunctions[id_button] = func;
}

function toggleSingleTokenCountingVisibility(id, id_counter, id_button) {
    var counter = gradioApp().getElementById(id_counter);
    var shouldDisplay = !opts.disable_token_counters;

    counter.style.display = shouldDisplay ? "block" : "none";
    counter.classList.toggle("token-counter-visible", shouldDisplay);
}

function toggleDualTokenCountingVisibility(id, id_t5_counter, id_clip_counter, id_button) {
    var t5_counter = gradioApp().getElementById(id_t5_counter);
    var clip_counter = gradioApp().getElementById(id_clip_counter);
    var shouldDisplay = !opts.disable_token_counters;

    t5_counter.style.display = shouldDisplay ? "block" : "none";
    clip_counter.style.display = shouldDisplay ? "block" : "none";

    t5_counter.classList.toggle("token-counter-visible", shouldDisplay);
    clip_counter.classList.toggle("token-counter-visible", shouldDisplay);
}

function runCodeForSingleTokenCounters(fun) {
    fun('txt2img_neg_prompt', 'txt2img_negative_token_counter', 'txt2img_negative_token_button');
    fun('img2img_neg_prompt', 'img2img_negative_token_counter', 'img2img_negative_token_button');
}

function runCodeForDualTokenCounters(fun) {
    fun('txt2img_prompt', 'txt2img_t5_token_counter', 'txt2img_token_counter', 'txt2img_token_button');
    fun('img2img_prompt', 'img2img_t5_token_counter', 'img2img_token_counter', 'img2img_token_button');
}

onUiLoaded(function() {
    runCodeForSingleTokenCounters(setupSingleTokenCounting);
    runCodeForDualTokenCounters(setupDualTokenCounting);
});

onOptionsChanged(function() {
    runCodeForSingleTokenCounters(toggleSingleTokenCountingVisibility);
    runCodeForDualTokenCounters(toggleDualTokenCountingVisibility);
});
