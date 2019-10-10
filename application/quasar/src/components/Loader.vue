<template>
<div :class="{
    ui: true,
    active: show === true ? true : false,
    inverted: inverted === true ? true : false,
    dimmer: true,
}">
    <div v-html="message.replace(/\n/gi, '<br />')" class="ui text loader">
    </div>
</div>
</template>

<script>
const DEFAULT_MESSAGE = '데이터를 가져오는 중입니다';

export default {
    name: 'Loader',

    props: {
        message: {
            type: String,
            default: DEFAULT_MESSAGE
        },

        show: {
            type: Boolean,
            default: false,
        },

        inverted: {
            type: Boolean,
            default: true,
        }
    },

    watch: {
        message: function(value) {
            if(!value)  this.message = DEFAULT_MESSAGE;
        }
    }
};
</script>

<style scoped style="less">
.ui.active.inverted.dimmer {
    z-index:99;
}
</style>
