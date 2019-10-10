<template>
<div v-ui-modal="{
        el: 'alertModal',
        setting: {
            duration: 300,

            onHide: onHide
        }
    }"
    class="ui mini modal">
    <i class="icon close"></i>
    <div class="header">
        {{title || '알림'}}
    </div>
    <div v-html="message.replace(/\n/gi, '<br />')" class="content"></div>
</div>
</template>

<script>
export default {
    name: 'AlertModal',

    props: {
        title: {
            type: String,
            default: '알림',
        },
        message: {
            type: String,
            default: '',
        },
        action: {
            type: String,
            default: 'hide',
        }
    },

    data()  {
        return {
            alertModal: null,
        };
    },

    methods: {
        onHide()    {
            this.$emit('update:action', 'hide');
            this.$emit('update:title', '알림');
            this.$emit('update:message', '');
        },
    },

    watch: {
        action: function()  {
            switch(this.action)  {
                case 'show':
                case 'hide':
                    this.alertModal[this.action]();
                    break;
                default:
                    this.alertModal.hide();
                    break;
            }
        },
    }
}
</script>
