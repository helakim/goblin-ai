<template>
<div class="ui fluid container main-wrap">
    <section class="ui grid container four column row compare-keywords-wrap">
        <div class="row">
            <div v-for="(a, index) in Array.from({ length: 4 })"
                class="column">
                <div class="ui search">
                    <i :class="'icon circle ' + $semanticColorsName[index]"></i>
                    <div class="ui big icon input">
                        <input
                            v-model.lazy.trim="queries[index]"
                            type="text" placeholder="검색어" />
                        <i class="icon grey close"></i>
                    </div>
                    <div class="results"></div>
                </div>
            </div>
        </div>
    </section>
    <section class="ui grid container one column row">
        <div class="column">
            <InterestTimeChart
                :queries="queries"
                :trendFlow="trendFlow"
                :loaderVisible="loader.interestTimeChart" />
        </div>
    </section>
    <section
        v-for="(query, index) in queries"
        class="ui grid container one column row">
        <div class="column">
            <RelatedQuery
                :query="query"
                :phrase="relatedPhrase[query] || []"
                :keywords="relatedKeywords[query] || []"
                :color="$semanticColorsName[index]"
                :loaderVisible="loader.relatedQuery"
                @update:expandView="action => onExpand(action, index)"
                @update:expandViewCategory="category => expandViewCategory = category" />
        </div>
    </section>

    <div v-for="(query, index) in queries"
        v-ui-modal="{
            el: 'expandViewModal' + index,
            setting: {
                duration: 300,
            }
        }"
        class="ui large modal">
        <i class="icon close"></i>
        <div class="header">
            <i :class="'icon circle ' + $semanticColorsName[index]"></i>
            {{query}}
        </div>
        <div class="content">
            <RelatedQuery
                :query="query"
                :phrase="relatedPhrase[query] || []"
                :keywords="relatedKeywords[query] || []"
                :color="$semanticColorsName[index]"
                :isModal="true"
                :expandViewCategory="expandViewCategory"
                :loaderVisible="loader.relatedQuery" />
        </div>
    </div>

    <div v-fixed class="fixed-keywords-wrap">
        <div v-show="isShowFixedKeywords" class="ui huge menu">
            <div v-for="(query, index) in queries"
                v-if="query"
                class="item">
                <i :class="'icon circle ' + $semanticColorsName[index]"></i>
                {{query}}
            </div>
        </div>
    </div>
</div>
</template>

<script>
import service from './service';
import InterestTimeChart from './InterestTimeChart';
import RelatedQuery from './RelatedQuery';


export default {
    name: 'TrendCompareResult',

    data()  {
        return {
            queries: [],
            queryForRoute: [],
            trendFlow: {},
            relatedPhrase: {},
            relatedKeywords: {},

            expandViewModal0: null,
            expandViewModal1: null,
            expandViewModal2: null,
            expandViewModal3: null,
            expandViewCategory: 'all',

            loader: {
                interestTimeChart: false,
                relatedQuery: false,
            },

            isShowFixedKeywords: false,
        };
    },

    watch: {
        $route: function(from, to)  {
            if(from.params.query !== to.params.query)   {
                let queries = this.$route.params.query
                    .replace(/^,{1,}/gi, '')
                    .replace(/\,{2,}/gi, ',')
                    .replace(/\,$/, '')
                    .split(',');

                this.getTrendCompare(queries);

                this.queries = queries;
            }
        },

        queries: function()   {
            this.$router.push(`/TrendCompare/${this.queries.join(',')}`)
        },

        scrollTop: function()   {
            this.scrollTop > 180 ? this.isShowFixedKeywords = true : this.isShowFixedKeywords = false;
        },
    },

    computed: {},

    methods: {
        async getTrendCompare(queries = [])   {
            let trendFlow = {}, relatedPhrase = {}, relatedKeywords = {};

            this.loader.interestTimeChart = this.loader.relatedQuery = true;

            this.trendFlow = {};
            this.relatedPhrase = {};
            this.relatedKeywords = {};

            for(let index = 0; index < queries.length; index++)  {
                let query = queries[index];

                if(!query || (query.constructor === String && query.length === 0))  {
                    // TODO:
                } else  {
                    let result = await service.getTrendCompare(query);

                    trendFlow[query] = result.trendflow.map(row => {
                        return { key: row.key, doc_count: row.doc_count };
                    });

                    relatedPhrase[query] = result.phrase.map(row => {
                        return {
                            key: row._source.doc,
                            score: row._score,
                        };
                    });

                    relatedKeywords[query] = result.keyword.slice(1, 11).map(row => {
                        return {
                            key: row.key,
                            doc_count: row.doc_count,
                        };
                    });
                }
            }

            this.trendFlow = trendFlow;
            this.relatedPhrase = relatedPhrase;
            this.relatedKeywords = relatedKeywords;

            this.loader.interestTimeChart = this.loader.relatedQuery = false;
        },

        onExpand(action, index)  {
            this['expandViewModal' + index][action]();
        },
    },

    components: {
        InterestTimeChart,
        RelatedQuery,
    },

    directives: {},

    beforeCreate()  {},
    created()   {
        this.queries = [];
        this.queries = this.$route.params.query
            .replace(/^,{1,}/gi, '')
            .replace(/\,{2,}/gi, ',')
            .replace(/\,$/, '')
            .split(',');

        this.getTrendCompare(this.queries);
    },
    beforeMount()   {},
    mounted()   {},
    beforeUpdate()  {},
    updated()   {},
    beforeDestroy() {},
    destroyed() {},
}
</script>

<style scoped lang="less">
.main-wrap {
    padding-top: 14px;
}

.compare-keywords-wrap {
    &.ui.grid.container.four.column.row {
        margin-left:0 !important;
        margin-right:0 !important;
        width:100% !important;
        box-shadow: 0 3px 4px lighten(grey, 30%);

        & > .row {
            margin-left: auto;
            margin-right: auto;
            padding-top: 0;
            padding-bottom: 0;
            width:1200px !important;

            & > .column {
                display: flex;
                padding: 0;
                height: 120px;
                border: 1px solid #dadada;
                border-right-width:0;
                border-top:0;
                border-bottom:0;
                align-items: center;

                .ui.search {
                    position: relative;
                    flex: 1;

                    & > i.icon.circle {
                        position: absolute;
                        left: 1em;
                        top: .9em;
                        z-index:1;
                    }

                    input {
                        padding-left: 2em;
                        border-radius: 0;
                        border-width: 0;

                        & + .remove, & + .close { visibility: hidden; }

                        &:focus {
                            // & + .remove, & + .close { visibility: visible; }
                        }
                    }
                }

                &:last-child {
                    border-right-width: 1px;
                }
            }
        }
    }
}

.fixed-keywords-wrap {
    position: fixed;
    top:0;
    left:0;
    right: 0;
    padding-left: 60px;
    z-index:102;

    .ui.menu {
        margin: 0 auto;

        .item {
            width:25%;
            text-align: center;
        }
    }
}
</style>
