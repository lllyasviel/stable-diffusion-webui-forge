// Monitors the gallery and sends a browser notification when the leading image is new.

let lastHeadImg = null;
let notificationButton = null;
let notificationPermission = Notification.permission; // Cache the initial permission state

function handleNotificationClick() {
    parent.focus();
    this.close();
}

function showNotification(headImgSrc, imageCount, returnGridSetting) {
    const actualImageCount = imageCount > 1 ? imageCount - returnGridSetting : 1;
    const notification = new Notification(
        'Stable Diffusion',
        {
            body: `Generated ${actualImageCount} image${actualImageCount > 1 ? 's' : ''}`,
            icon: headImgSrc,
            image: headImgSrc,
        }
    );
    notification.onclick = handleNotificationClick;
}

function playNotificationSound() {
    const notificationAudio = gradioApp().querySelector('#audio_notification #waveform > div')?.shadowRoot?.querySelector('audio');
    if (notificationAudio) {
        notificationAudio.volume = opts?.notification_volume / 100.0 || 1.0;
        notificationAudio.play();
    }
}

function checkAndUpdateGallery() {
    const galleryPreviews = gradioApp().querySelectorAll('div[id^="tab_"] div[id$="_results"] .thumbnail-item > img');
    if (!galleryPreviews || galleryPreviews.length === 0) return;

    const headImgSrc = galleryPreviews[0]?.src;
    if (!headImgSrc || headImgSrc === lastHeadImg) return;

    lastHeadImg = headImgSrc;

    playNotificationSound();

    if (!document.hasFocus() && notificationPermission === 'granted') {
        const imageCount = new Set(Array.from(galleryPreviews).map(img => img.src)).size;
        showNotification(headImgSrc, imageCount, opts?.return_grid || 0);
    }
}

onAfterUiUpdate(function() {
    // Initialize the notification button and its listener only once
    if (!notificationButton) {
        notificationButton = gradioApp().getElementById('request_notifications');
        if (notificationButton) {
            notificationButton.addEventListener('click', () => {
                Notification.requestPermission().then(permission => {
                    notificationPermission = permission; // Update cached permission
                });
            }, true);
        }
    }

    checkAndUpdateGallery();
});
