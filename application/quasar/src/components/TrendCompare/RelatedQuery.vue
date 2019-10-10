<template>
<div class="ui fluid container">
    <div v-if="!isModal" class="ui inverted attached top header">
        <i :class="'icon mini circle ' + color"></i>
        <div class="content">
            <h3>{{query}}</h3>
        </div>
    </div>
    <div class="ui attached segment">
        <div class="ui internally celled grid">
            <div
                v-if="expandViewCategory === 'all' || expandViewCategory === phraseTitle"
                :class="{
                    sixteen: expandViewCategory === phraseTitle ? true : false,
                    eight: expandViewCategory === 'all' ? true : false,
                    wide: true,
                    column: true,
                }">
                <div class="ui borderless menu">
                    <h3 class="item header">
                        {{phraseTitle}}
                    </h3>
                    <div v-if="!isModal" class="menu right">
                        <div class="item">
                            <div
                                @click="onExpand(phraseTitle)"
                                class="ui basic icon button">
                                <i class="icon expand"></i>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="ui large middle aligned divided list">
                    <div v-for="(item, index) in phrase"
                        class="item">
                        <div class="left floated content">
                            {{item.key}}
                        </div>
                        <div class="right floated content">
                            {{(item.score / phraseTopScore * 100).toFixed(0)}}
                            <div v-ui-progress="{
                                    setting: {
                                        percent: item.score / phraseTopScore * 100,
                                        className: {
                                            success: '100'
                                        }
                                    }
                                }"
                                :class="'ui small progress ' + color">
                                <div class="bar"></div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            <div
                v-if="expandViewCategory === 'all' || expandViewCategory === keywordTitle"
                :class="{
                    sixteen: expandViewCategory === keywordTitle ? true : false,
                    eight: expandViewCategory === 'all' ? true : false,
                    wide: true,
                    column: true,
                }">
                <div class="ui borderless menu">
                    <h3 class="item header">
                        {{keywordTitle}}
                    </h3>
                    <div v-if="!isModal" class="menu right">
                        <div class="item">
                            <div
                                @click="onExpand(keywordTitle)"
                                class="ui basic icon button">
                                <i class="icon expand"></i>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="ui large middle aligned divided list">
                    <div v-for="(item, index) in keywords"
                        class="item">
                        <div class="left floated content">
                            {{item.key}}
                        </div>
                        <div class="right floated content">
                            {{(item.doc_count / keywordsTopScore * 100).toFixed(0)}}
                            <div v-ui-progress="{
                                    setting: {
                                        percent: item.doc_count / keywordsTopScore * 100,
                                        className: {
                                            success: '100'
                                        }
                                    }
                                }"
                                :class="'ui small progress ' + color">
                                <div class="bar"></div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        <Loader :show="loaderVisible" />
    </div>
</div>
</template>

<script>
export default {
    name: 'TrendCompareResultRelatedQuery',

    props: {
        loaderVisible: {
            type: Boolean,
            default: false,
        },

        isModal: {
            type: Boolean,
            default: false,
        },

        expandViewCategory: {
            type: String,
            default: 'all',
        },

        color: {
            type: String,
            default: '',
        },

        query: {
            type: String,
            default: '',
        },

        phrase: {
            type: Array,
            default: [],
        },

        keywords: {
            type: Array,
            default: [],
        },
    },

    data()  {
        return {
            keywordTitle: '관련 검색어',
            phraseTitle: '관련 토픽',

            keywordsTopScore: 0,
            phraseTopScore: 0,
        };
    },

    watch: {
        keywords: function(value)    {
            if(value.length)    {
                this.keywordsTopScore = this.keywords[0].doc_count;
            }
        },

        phrase: function(value)    {
            if(value.length)    {
                this.phraseTopScore = this.phrase[0].score;
            }
        },
    },

    computed: {},

    methods: {
        onExpand(title)  {
            this.$emit('update:expandView', 'show');
            this.$emit('update:expandViewCategory', title);
        }
    },

    components: {},

    directives: {},

    beforeCreate()  {},
    created()   {},
    beforeMount()   {},
    mounted()   {},
    beforeUpdate()  {},
    updated()   {},
    beforeDestroy() {},
    destroyed() {},
}
</script>

<style scoped lang="less">
.ui.list {
    min-height: 400px;

    .item {
        padding-top: 1em;
        padding-bottom: 1em;

        .floated.content {
            margin-left: 0;
            margin-right: 0;

            text-overflow: ellipsis;
            white-space: nowrap;
            overflow: hidden;
            max-width:70%;

            .ui.progress {
                display:inline-block;
                margin-left: .75em;
                margin-bottom: 0;
                width: 80px;
                vertical-align: middle;
                border-radius: 0;

                .bar {
                    border-radius: 0;
                }
            }
        }
    }
}

.ui.borderless.menu {
    border:0;
    box-shadow: 0 0 0 #ffffff;

    .item.header { padding-left:0; padding-right:0; }
}
</style>

