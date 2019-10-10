<template>
<div class="column">
    <div class="ui blue border header">
        <h3 class="ui container left aligned labels">
            텍스트 전환 결과

            <div v-if="!isModal" class="ui basic label">
                <i class="icon red square"></i>
            </div>
            <div v-if="!isModal" class="ui basic label">
                <i class="icon blue square"></i>
            </div>
        </h3>
    </div>
    <div class="ui container">
        <div v-if="!isModal" class="ui grid container three column row">
            <div class="column">
                <div class="ui toggle blue checkbox">
                    <input id="follow-scroll" type="checkbox">
                    <label for="follow-scroll"></label>
                </div>
            </div>
            <div class="column">
                <div class="ui left icon small action input">
                    <input type="text" name="text" placeholder="찾기 (입력 후 엔터)" />
                    <i class="search icon"></i>
                    <button class="ui blue small icon button">
                        <i class="chevron left icon"></i>
                    </button>
                    <button class="ui blue small icon button">
                        <i class="chevron right icon"></i>
                    </button>
                </div>

            </div>
            <div class="column">
                <div class="ui container right aligned">
                    <div class="ui blue small icon buttons">
                        <button class="ui icon button">
                            <i class="window maximize icon"></i>
                        </button>
                        <button class="ui button">
                            <i class="plus icon"></i>
                        </button>
                        <button class="ui button">
                            <i class="minus icon"></i>
                        </button>
                    </div>
                </div>
            </div>
        </div>
    </div>
    <div class="ui divider"></div>
    <div class="ui segment">
        <div class="ui container scripts-warp">
            <div class="ui list">
                <div
                    v-for="script in retrieved"
                    v-show="script.trim().length"
                    class="item">
                    <i class="icon caret right"></i>
                    <p>
                        {{script.trim()}}
                    </p>
                </div>
            </div>
        </div>
        <Loader :show="loaderVisible" />
    </div>
    <div class="ui divider"></div>
</div>
</template>

<script>
import searchService from '../Search/service';


export default {
    name: 'Scripts',

    props: {
        isModal: {
            type: Boolean,
            default: false,
        },

        loaderVisible: {
            type: Boolean,
            default: false,
        },

        scripts: {
            type: Array,
            default: [],
        }
    },

    data()  {
        return {
            retrieved: [],
        }
    },

    methods: {

    },

    computed: {

    },

    watch: {
        scripts: function() {
            this.retrieved = [];

            this.scripts
                .map(script => script.retrieved.split(/\n/))
                // .map(script => script.retrieved)
                .forEach(retrieved => {
                this.retrieved = this.retrieved.concat(retrieved);
            });
            // this.retrieved = this.scripts.map(script => script.retrieved);
        },
    },

    directives: {

    },

    components: {

    },

    beforeCreate()  {},
	created()	{},
    beforeMount()   {},
	mounted() 	{},
    beforeUpdate()  {},
    updated()   {},
    beforeDestroy() {},
    destroyed() {}
};

</script>

<style scoped lang="less">
.scripts-warp {
    height:372px;min-height:372px;max-height:372px;overflow-y:scroll;

    .ui.list {
        .item {
            position: relative;

            i.icon {
                position: absolute;
                top:5px;left:0;
            }

            p {
                margin-left:14px;
            }

            &:first-child {
                i.icon { top: 1px; }
            }
        }
    }
}
</style>
