
var contextMenuInit = function() {
    let eventListenerApplied = false;
    let menuSpecs = new Map();

    const uid = function() {
        return Date.now().toString(36) + Math.random().toString(36).substring(2);
    };

    function showContextMenu(event, element, menuEntries) {
        let oldMenu = gradioApp().querySelector('#context-menu');
        if (oldMenu) {
            oldMenu.remove();
        }

        let baseStyle = window.getComputedStyle(uiCurrentTab);

        const contextMenu = document.createElement('nav');
        contextMenu.id = "context-menu";
        contextMenu.style.background = baseStyle.background;
        contextMenu.style.color = baseStyle.color;
        contextMenu.style.fontFamily = baseStyle.fontFamily;
        contextMenu.style.top = event.pageY + 'px';
        contextMenu.style.left = event.pageX + 'px';

        const contextMenuList = document.createElement('ul');
        contextMenuList.className = 'context-menu-items';
        contextMenu.append(contextMenuList);

        menuEntries.forEach(function(entry) {
            let contextMenuEntry = document.createElement('a');
            contextMenuEntry.innerHTML = entry['name'];
            contextMenuEntry.addEventListener("click", function() {
                entry['func']();
            });
            contextMenuList.append(contextMenuEntry);

        });

        gradioApp().appendChild(contextMenu);
    }

    function appendContextMenuOption(targetElementSelector, entryName, entryFunction) {

        var currentItems = menuSpecs.get(targetElementSelector);

        if (!currentItems) {
            currentItems = [];
            menuSpecs.set(targetElementSelector, currentItems);
        }
        let newItem = {
            id: targetElementSelector + '_' + uid(),
            name: entryName,
            func: entryFunction,
            isNew: true
        };

        currentItems.push(newItem);
        return newItem['id'];
    }

    function removeContextMenuOption(uid) {
        menuSpecs.forEach(function(v) {
            let index = -1;
            v.forEach(function(e, ei) {
                if (e['id'] == uid) {
                    index = ei;
                }
            });
            if (index >= 0) {
                v.splice(index, 1);
            }
        });
    }

    function addContextMenuEventListener() {
        if (eventListenerApplied) {
            return;
        }
        gradioApp().addEventListener("click", function(e) {
            if (!e.isTrusted) {
                return;
            }

            let oldMenu = gradioApp().querySelector('#context-menu');
            if (oldMenu) {
                oldMenu.remove();
            }
        });
        ['contextmenu', 'touchstart'].forEach((eventType) => {
            gradioApp().addEventListener(eventType, function(e) {
                let ev = e;
                if (eventType.startsWith('touch')) {
                    if (e.touches.length !== 2) return;
                    ev = e.touches[0];
                }
                let oldMenu = gradioApp().querySelector('#context-menu');
                if (oldMenu) {
                    oldMenu.remove();
                }
                menuSpecs.forEach(function(v, k) {
                    if (e.composedPath()[0].matches(k)) {
                        showContextMenu(ev, e.composedPath()[0], v);
                        e.preventDefault();
                    }
                });
            });
        });
        eventListenerApplied = true;

    }

    return [appendContextMenuOption, removeContextMenuOption, addContextMenuEventListener];
};

var initResponse = contextMenuInit();
var appendContextMenuOption = initResponse[0];
var removeContextMenuOption = initResponse[1];
var addContextMenuEventListener = initResponse[2];

var regen_txt2img = null;
var regen_img2img = null;

(function() {
    //Start example Context Menu Items
    let generateOnRepeat_txt2img = function() {
		if ((regen_txt2img == null) && (regen_img2img == null)) {
			let generate = gradioApp().querySelector('#txt2img_generate');
			let interrupt = gradioApp().querySelector('#txt2img_interrupt');
			if (!interrupt.offsetParent) {
				generate.click();
			}

			regen_txt2img = setInterval(function() {
				if (interrupt.style.display == 'none') {
					generate.click();
					interrupt.style.display = 'block';
				}
			},
			500);
		}
    };
    appendContextMenuOption('#txt2img_generate', 'Generate forever', generateOnRepeat_txt2img);
    appendContextMenuOption('#txt2img_interrupt', 'Generate forever', generateOnRepeat_txt2img);

    let cancel_regen_txt2img = function() {
        clearInterval(regen_txt2img);
		regen_txt2img = null;
    };
    appendContextMenuOption('#txt2img_interrupt', 'Cancel generate forever', cancel_regen_txt2img);
    appendContextMenuOption('#txt2img_generate', 'Cancel generate forever', cancel_regen_txt2img);

    let generateOnRepeat_img2img = function() {
		if ((regen_txt2img == null) && (regen_img2img == null)) {
			let generate = gradioApp().querySelector('#img2img_generate');
			let interrupt = gradioApp().querySelector('#img2img_interrupt');
			if (!interrupt.offsetParent) {
				generate.click();
			}

			regen_img2img = setInterval(function() {
				if (interrupt.style.display == 'none') {
					generate.click();
					interrupt.style.display = 'block';
				}
			},
			500);
		}
    };
    appendContextMenuOption('#img2img_generate', 'Generate forever', generateOnRepeat_img2img);
    appendContextMenuOption('#img2img_interrupt', 'Generate forever', generateOnRepeat_img2img);

    let cancel_regen_img2img = function() {
        clearInterval(regen_img2img);
		regen_img2img = null;
    };
    appendContextMenuOption('#img2img_interrupt', 'Cancel generate forever', cancel_regen_img2img);
    appendContextMenuOption('#img2img_generate', 'Cancel generate forever', cancel_regen_img2img);

})();


onAfterUiUpdate(addContextMenuEventListener);
