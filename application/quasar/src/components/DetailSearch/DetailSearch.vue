<template>
<div v-ui-modal="{
        el: 'modal',
        setting: {
            duration: 300,
            onHide: function()  {
                $emit('update:visible', false);
            }
        },
    }"
    class="ui modal">
    <i class="icon close"></i>
    <div class="header">
        고급검색
    </div>
    <div class="contents">
        <div class="ui basic segment">
            <div class="ui grid container">
                <div class="ui column row grid">
                    <div class="five wide column middle aligned">
                        {{ labelName.base }}
                    </div>
                    <div class="eleven wide column">
                        <div class="ui input fluid">
                            <input v-model="detailForm.base" type="text" />
                        </div>
                    </div>
                </div>
                <div class="ui column row grid">
                    <div class="five wide column middle aligned">
                        {{ labelName.must }}
                    </div>
                    <div class="eleven wide column">
                        <div class="ui input fluid">
                            <input v-model="detailForm.must" type="text" />
                        </div>
                    </div>
                </div>
                <div class="ui column row grid">
                    <div class="five wide column middle aligned">
                        {{ labelName.or }}
                    </div>
                    <div class="eleven wide column">
                        <div class="ui input fluid">
                            <input v-model="detailForm.or" type="text" />
                        </div>
                    </div>
                </div>
                <div class="ui column row grid">
                    <div class="five wide column middle aligned">
                        {{ labelName.not }}
                    </div>
                    <div class="eleven wide column">
                        <div class="ui input fluid">
                            <input v-model="detailForm.not" type="text" />
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    <div class="actions">
        <button @click="close" class="ui blue button">
            고급검색
        </button>
    </div>
</div>
</template>

<script>
import { DetailForm } from './model';

export default {
    name: 'DetailSearch',

    props: {
        visible: {
            type: Boolean,
            default: false
        },

        searchValue: {
            type: String,
            default: '',
        },
    },

    data()  {
        return {
            modal: null,

            detailForm: new DetailForm,

            labelName: {
                base: '다음 단어 모두 포함',
                must: '다음 단어 또는 문구 정확하게 포함',
                or: '다음 단어 중 아무거나 포함',
                not: '다음 단어 제외',
            }
        };
    },

    watch: {
        visible: function() {
            if(this.visible)    this.modal.show();
        },

        searchValue: function(value) {
            if(value && value.length)   {
                this.detailForm = DetailForm.parse(value);

                this.$emit('update:detailFormString', this.detailForm.toString());
                this.$emit('update:detailForm', this.detailForm.toJSON());
            }
        },
    },

    methods: {
        close() {
            this.$emit('update:detailFormString', this.detailForm.toString());
            this.$emit('update:detailForm', this.detailForm.toJSON());
            this.modal.hide();
        },
    },

    mounted()   {

    },
};
</script>

<style lang="less" scoped>

</style>
