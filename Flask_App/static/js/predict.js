
$(document).ready(function () {

    const $valueSpan = $('.on_kamera');
    const $value = $('#customRange11');
    $valueSpan.html($value.val());
    $value.on('input change', () => {

        $valueSpan.html($value.val());
    });
});