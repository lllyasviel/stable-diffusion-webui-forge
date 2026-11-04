"""Secure Presets: encrypted, passphrase-protected prompt/settings presets for txt2img and img2img.

Presets (prompts, negative prompts, sampler/size/CFG/seed settings, hires or denoising settings and
the "UI" radio choice) are stored only as AES-256-GCM ciphertext in secure_presets/vault.json.
One library is shared by both tabs: loading a preset applies every field that exists on the current tab.
See modules/secure_store.py for the storage and crypto details.
"""
import os

import gradio as gr
from gradio.context import Context

from modules import scripts, script_callbacks, secure_store, shared

# (preset key, elem_id) -- "{t}" is replaced by the tab name
COMMON_FIELDS = [
    ("prompt", "{t}_prompt"),
    ("negative_prompt", "{t}_neg_prompt"),
    ("sampler", "{t}_sampling"),
    ("scheduler", "{t}_scheduler"),
    ("steps", "{t}_steps"),
    ("width", "{t}_width"),
    ("height", "{t}_height"),
    ("batch_count", "{t}_batch_count"),
    ("batch_size", "{t}_batch_size"),
    ("cfg_scale", "{t}_cfg_scale"),
    ("distilled_cfg_scale", "{t}_distilled_cfg_scale"),
    ("seed", "{t}_seed"),
]
TAB_FIELDS = {
    "txt2img": [
        ("hires_prompt", "hires_prompt"),
        ("hires_negative_prompt", "hires_neg_prompt"),
        ("hires_enable", "txt2img_hr-checkbox"),
        ("hires_upscaler", "txt2img_hr_upscaler"),
        ("hires_steps", "txt2img_hires_steps"),
        ("hires_denoise", "txt2img_denoising_strength"),
        ("hires_scale", "txt2img_hr_scale"),
        ("hires_resize_x", "txt2img_hr_resize_x"),
        ("hires_resize_y", "txt2img_hr_resize_y"),
        ("hires_cfg", "txt2img_hr_cfg"),
        ("hires_distilled_cfg", "txt2img_hr_distilled_cfg"),
        ("hires_sampler", "hr_sampler"),
        ("hires_scheduler", "hr_scheduler"),
    ],
    "img2img": [
        ("denoising_strength", "img2img_denoising_strength"),
        ("resize_scale", "img2img_scale"),
    ],
}
UI_RADIO_ELEM_ID = "forge_ui_preset"   # the sd / xl / flux / all radio at the top of the page


def fields_for(tab):
    return [(k, e.format(t=tab)) for k, e in COMMON_FIELDS] + TAB_FIELDS[tab]


_components = {}
_pending = {}
_panels = {}
_KEEP = object()
_WANTED = {UI_RADIO_ELEM_ID} | {e for tab in TAB_FIELDS for _, e in fields_for(tab)}


def _capture(component, **kwargs):
    elem_id = kwargs.get("elem_id")
    # Only real input components: layouts such as an InputAccordion's Accordion share the same elem_id.
    if elem_id in _WANTED and isinstance(component, gr.components.Component):
        _components[elem_id] = component


script_callbacks.on_after_component(_capture)

# The "read last generation" feature (and its params.txt / params.enc files) was removed.
# Clean up leftovers: overwrite with zeros and delete, never read.
try:
    from modules import paths
    for _legacy in (os.path.join(paths.data_path, "params.txt"),
                    os.path.join(paths.data_path, "secure_presets", "params.enc")):
        if secure_store.wipe_plaintext(_legacy):
            print(f"Secure Presets: removed leftover {os.path.basename(_legacy)}")
except Exception:
    pass


class SecurePresetsScript(scripts.Script):
    sorting_priority = 9000

    def title(self):
        return "Secure Presets"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        tab = "img2img" if is_img2img else "txt2img"
        if tab == "txt2img":  # first tab built: forget panels from any previous UI build (UI reload)
            _panels.clear()
            _pending.clear()

        present = [(k, e) for k, e in fields_for(tab) if e in _components]
        missing = [k for k, e in fields_for(tab) if e not in _components]
        if missing:
            print(f"Secure Presets ({tab}): fields not found in this UI version, skipped: {', '.join(missing)}")
        keys = [k for k, _ in present]
        field_comps = [_components[e] for _, e in present]
        radio = _components.get(UI_RADIO_ELEM_ID)

        def view(msg="", selected=None):
            if not secure_store.crypto_available():
                return [gr.update(value="⚠️ The 'cryptography' package is missing. Restart Forge so it can install it."),
                        gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update()]
            exists, unlocked = secure_store.vault_exists(), secure_store.is_unlocked()
            names = secure_store.list_names() if unlocked else []
            if not exists:
                state = ("No vault yet. Choose a passphrase of at least 12 characters. "
                         "**There is no way to recover it: if you forget it, the presets are gone.**")
            elif not unlocked:
                state = "🔒 Locked. Enter your passphrase to use presets."
            else:
                state = f"🔓 Unlocked ({len(names)} preset{'s' if len(names) != 1 else ''})."
            text = f"{state}\n\n{msg}" if msg else state
            if selected is _KEEP:
                dd_update = gr.update(choices=names)
            else:
                dd_update = gr.update(choices=names, value=selected if selected in names else None)
            return [gr.update(value=text),
                    gr.update(visible=not exists),
                    gr.update(visible=exists and not unlocked),
                    gr.update(visible=unlocked),
                    dd_update]

        def refresh():
            try:
                return view(selected=_KEEP)
            except Exception:
                return view("⚠️ Could not refresh.", selected=_KEEP)

        def guarded(extra=0):
            """Wrap a handler so errors become a status message; pads to the handler's output count."""
            def decorator(fn):
                def wrapper(*args):
                    pad = [gr.update() for _ in range(extra)]
                    try:
                        return fn(*args)
                    except secure_store.VaultError as e:
                        return view(f"⚠️ {e}") + pad
                    except Exception as e:
                        print(f"Secure Presets error: {type(e).__name__}")
                        return view("⚠️ Unexpected error (details withheld from the log to avoid leaking text).") + pad
                return wrapper
            return decorator

        with gr.Accordion("Secure Presets 🔒", open=False, elem_id=f"secure_presets_{tab}"):
            status = gr.Markdown()

            with gr.Group(visible=False) as g_setup:
                pw_new = gr.Textbox(label="New passphrase (min 12 characters)", type="password")
                pw_new2 = gr.Textbox(label="Repeat passphrase", type="password")
                btn_create = gr.Button("Create encrypted vault", variant="primary")

            with gr.Group(visible=False) as g_unlock:
                pw_unlock = gr.Textbox(label="Passphrase", type="password")
                btn_unlock = gr.Button("Unlock", variant="primary")

            with gr.Group(visible=False) as g_main:
                dd = gr.Dropdown(label="Saved presets", choices=[], value=None, interactive=True)
                with gr.Row():
                    btn_load = gr.Button("Load preset", variant="primary")
                    btn_lock = gr.Button("Lock")
                name_tb = gr.Textbox(label="Preset name (saving an existing name overwrites it)", max_lines=1)
                btn_save = gr.Button("Save current prompts + settings as preset")
                with gr.Row():
                    chk_del = gr.Checkbox(label="Confirm delete", value=False)
                    btn_del = gr.Button("Delete selected preset")
                with gr.Accordion("Change passphrase", open=False):
                    pw_old = gr.Textbox(label="Current passphrase", type="password")
                    pw_chg = gr.Textbox(label="New passphrase (min 12 characters)", type="password")
                    btn_chg = gr.Button("Change passphrase")

        view_out = [status, g_setup, g_unlock, g_main, dd]

        @guarded(2)
        def do_create(p1, p2):
            if p1 != p2:
                return view("⚠️ The two passphrases do not match.") + [gr.update(), gr.update()]
            secure_store.create_vault(p1)
            return view("Vault created and unlocked.") + [gr.update(value=""), gr.update(value="")]

        @guarded(1)
        def do_unlock(p):
            secure_store.unlock(p)
            return view("Unlocked.") + [gr.update(value="")]

        def do_lock():
            secure_store.lock()
            return view("Locked.")

        @guarded(0)
        def do_save(name, *vals):
            ui_choice = vals[-1] if radio is not None else None
            values = dict(zip(keys, vals[:len(keys)]))
            saved = secure_store.save_preset(name, ui_choice, values, tab=tab)
            return view(f"Saved preset “{saved}”.", selected=saved)

        @guarded(1)
        def do_delete(sel, confirm):
            if not sel:
                return view("Select a preset first.") + [gr.update()]
            if not confirm:
                return view("Tick “Confirm delete” first.", selected=sel) + [gr.update()]
            secure_store.delete_preset(sel)
            return view(f"Deleted preset “{sel}”.") + [gr.update(value=False)]

        @guarded(2)
        def do_change(old, new):
            secure_store.change_passphrase(old, new)
            return view("Passphrase changed.") + [gr.update(value=""), gr.update(value="")]

        def do_load_prepare(sel):
            # Step 1: switch the UI radio first. Forge's own handler for that radio resets size/CFG/sampler,
            # so the remaining settings are applied in step 2, after it has finished.
            lead = [gr.update()] if radio is not None else []
            try:
                preset = secure_store.get_preset(sel) if sel else None
                if preset is None:
                    _pending[tab] = None
                    return lead + [gr.update(value="⚠️ Select a preset first.")]
                _pending[tab] = (sel, preset.get("values", {}))
                target = preset.get("ui")
                change = target and target != getattr(shared.opts, "forge_preset", None)
                out = [gr.update(value=target) if change else gr.update()] if radio is not None else []
                return out + [gr.update(value=f"Loading preset “{sel}”…")]
            except secure_store.VaultError as e:
                _pending[tab] = None
                return lead + [gr.update(value=f"⚠️ {e}")]

        def do_load_apply():
            pending = _pending.pop(tab, None)
            if pending is None:
                return [gr.update() for _ in field_comps] + [gr.update()]
            name, values = pending
            return [gr.update(value=values[k]) if k in values else gr.update() for k in keys] + [gr.update(value=f"🔓 Loaded preset “{name}”.")]

        deps = [
            btn_create.click(do_create, inputs=[pw_new, pw_new2], outputs=view_out + [pw_new, pw_new2], show_progress="minimal"),
            btn_unlock.click(do_unlock, inputs=[pw_unlock], outputs=view_out + [pw_unlock], show_progress="minimal"),
            btn_lock.click(do_lock, inputs=None, outputs=view_out, show_progress="hidden"),
            btn_save.click(do_save, inputs=[name_tb] + field_comps + ([radio] if radio is not None else []), outputs=view_out, show_progress="minimal"),
            btn_del.click(do_delete, inputs=[dd, chk_del], outputs=view_out + [chk_del], show_progress="hidden"),
            btn_chg.click(do_change, inputs=[pw_old, pw_chg], outputs=view_out + [pw_old, pw_chg], show_progress="minimal"),
        ]

        prepare_out = ([radio] if radio is not None else []) + [status]
        btn_load.click(do_load_prepare, inputs=[dd], outputs=prepare_out, show_progress="hidden", queue=False).then(
            do_load_apply, inputs=None, outputs=field_comps + [status], show_progress="hidden", queue=False,
            js="async () => { await new Promise(r => setTimeout(r, 800)); }",
        )

        Context.root_block.load(lambda: view(), inputs=None, outputs=view_out, queue=False, show_progress="hidden")

        # Keep the txt2img and img2img panels in step (lock state, preset list) without a page reload.
        for other in _panels.values():
            for dep in other["deps"]:
                dep.then(refresh, inputs=None, outputs=view_out, show_progress="hidden", queue=False)
            for dep in deps:
                dep.then(other["refresh"], inputs=None, outputs=other["view_out"], show_progress="hidden", queue=False)
        _panels[tab] = {"deps": deps, "refresh": refresh, "view_out": view_out}
        return []
