<template>
<div class="ui middle aligned centered fluid grid container main-wrap">
    <section class="ui grid container one column row">
        <div class="seven wide column">
            <div class="ui basic segment no-padding container">
                <div class="ui centered medium image">
                    <img :src="require('../../assets/vuejs-logo.png')" alt="">
                </div>
                <div v-ui-search="{
                        el: 'search',
                        setting: {
                            minCharacters: 1,
                            apiSettings: {
                                url: '//localhost:18080/api/demo/trendcompare/{query}',
                                //onResponse: onResponse,
                            },
                            selector: {
                                prompt: 'input[type=text]'
                            },
                            onSelect: onSelect,
                            onResults: onSearch,
                            maxResults: 10,
                            showNoResults: false,
                        }
                    }"
                    class="ui search">
                    <div class="ui large fluid icon input">
                        <input v-focus
                            @keypress.enter="() => { $router.push(`/TrendCompare/${search.getValue()}`) }"
                            type="text" placeholder="">
                        <i class="search icon"></i>
                    </div>
                    <div class="results"></div>
                </div>
            </div>
        </div>
    </section>
    <router-view :searchResult="searchResult"></router-view>
</div>
</template>

<script>
export default {
    name: 'TrendCompare',

    data()  {
        return {
            search: null,
            searchResult: null,
        };
    },

    watch: {},

    computed: {},

    methods: {
        onSearch(res)  {
            console.log(res);
            this.searchResult = res;
        },

        onResponse(res)    {

            return res;
        },

        onSelect(result, res)  {
            console.log(result, res);
        },
    },

    components: {},

    directives: {},

    beforeCreate()  {},
    created()   {},
    beforeMount()   {},
    mounted()   {
        this.$el.style.height = `${this.getWindowHeight() - this.getAppTopMenuHeight()}px`;

        window.addEventListener('resize', () => {
            this.$el.style.height = `${this.getWindowHeight() - this.getAppTopMenuHeight()}px`;
        });
    },
    beforeUpdate()  {},
    updated()   {},
    beforeDestroy() {},
    destroyed() {},
}
</script>

<style scoped lang="less">
.main-wrap {
    margin:0;
    background-color:transparent;
}

.ui.grid.container.one.column.row { padding:0; }

.ui.input {
    input {
        border-radius:2px;
        box-shadow: 0 2px 3px lighten(grey, 40%);

        &:focus {
            box-shadow: 0 2px 3px lighten(grey, 30%);
        }
    }
}

.ui.search > .results {
    margin-top:0;
    width:100%;
    border-radius: 0;
    border-top-width:0;
}

.ui.centered.medium.image {
    display:block;
    margin-top:-30%;
    margin-bottom:1em;
}
</style>