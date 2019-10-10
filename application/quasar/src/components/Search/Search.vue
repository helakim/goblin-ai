<template>
<div class="ui fluid container main-wrap">
    <section class="ui internally celled grid one column row">
        <div class="ui gird internally celled column one row">
            <div class="column row search-wrap">
                <div class="ui labeled action input">
                    <div class="ui blue label">
                        키워드
                    </div>
                    <div class="ui icon input">
                        <i class="icon search"></i>
                        <input
                            v-focus
                            v-model.lazy="searchKeyword"
                            @keypress.enter="onSearch(0)"
                            type="text">
                    </div>
                    <button
                        @click="onSearch(0)"
                        class="ui blue button">조회</button>
                    <button
                        @click="visibleDetailSearch = true"
                        class="ui blue button">검색조건설정</button>
                </div>

                <a v-for="relationKeyword in relationKeywords"
                    @click="() => { searchKeyword = relationKeyword.key; onSearch(0); }"
                    class="ui basic icon label">
                    <i class="icon search"></i>
                    {{relationKeyword.key}}
                </a>
            </div>

            <div v-if="false" class="column">
                <div class="ui fluid container left aligned">
                    <div v-ui-dropdown="{
                            el: 'sortDropdown',
                            setting: {
                                onChange: onSortChange
                            }
                        }"
                        class="ui blue labeled icon button dropdown">
                        <i class="sort icon"></i>
                        <span class="text">정렬방식</span>
                        <div class="menu">
                            <div data-value="score" class="item">
                                점수순
                            </div>
                            <div data-value="frequency" class="item">
                                빈도순
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </section>
    <section class="ui internally celled middle centered aligned grid one column row">
        <div class="fifteen wide column">
            <div class="ui segment">
                <a class="ui big blue ribbon label">
                     # {{keyword}}
                </a>

                <div class="ui divider"></div>

                <div class="ui fluid container basic segment no-padding">
                    <div class="ui list">
                        <div
                            v-for="document in documents"
                            class="link item"
                            >
                            <div class="content">
                                <div class="description">
                                    <div class="ui info message">
                                        <div class="header">
                                            <small>점수: {{document._score}}</small>
                                        </div>
                                        <!-- <div class="doc-content" v-html="document.highlight.doc[0]"></div> -->
                                        <div class="doc-content" v-html="document._source.doc"></div>
                                        <p>
                                            <u @click="showScripts(document)">내용보기</u>
                                        </p>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div v-if="!documents.length" class="no-data link item">
                            <div class="content">
                                <div class="ui aligned center container">
                                    <i class="grey big warning circle icon"></i>
                                    <p>조회 된 데이터가 없습니다</p>
                                </div>
                            </div>
                        </div>
                    </div>
                    <Loader v-bind:show="loader.searchResult" />
                </div>

                <div
                    v-if="documents.length"
                    class="ui container center aligned">
                    <div class="ui link list huge horizontal page-wrap">
                        <a
                            @click="onSearch(0)"
                            class="item">
                            <i class="icon blue angle double left"></i>
                        </a>
                        <a
                            @click="onSearch((Math.floor(page.current / page.size) - 1) * page.size)"
                            class="item">
                            <i class="icon blue angle left"></i>
                        </a>
                        <a
                            v-for="no in page.list"
                            @click="onSearch(no)"
                            :class="{
                                item: true,
                                active: page.current === no
                            }">
                            {{no + 1}}
                        </a>
                        <a
                            @click="onSearch((Math.floor(page.current / page.size) + 1) * page.size)"
                            class="item">
                            <i class="icon blue angle right"></i>
                        </a>
                        <a
                            @click="onSearch(page.end)"
                            class="item">
                            <i class="icon blue angle double right"></i>
                        </a>
                    </div>
                </div>
            </div>
        </div>
    </section>

    <div
        v-ui-modal="{
            el: 'detailModal',
            setting: {
                duration: 300,
            }
        }"
        class="ui modal"
    >
        <div class="ui grid container one column row content">
            <div class="column">
                <div class="ui info message">
                    <div class="header">
                        <small>점수: {{selectedDocument._score}}</small>
                    </div>
                    <!-- <div class="doc-content" v-html="selectedDocument.highlight.doc[0]"></div> -->
                </div>
            </div>
            <Scripts
                :isModal="true"
                :scripts="scripts"
                :loaderVisible="loader.scripts"
            />
        </div>
    </div>

    <DetailSearch
        :visible="visibleDetailSearch"
        :searchValue="searchKeyword"
        @update:visible="visible => visibleDetailSearch = visible"
        @update:detailFormString="detailFormString => searchKeyword = detailFormString"
        @update:detailForm="_detailForm => detailForm = _detailForm" />
</div>
</template>

<script>
import Router from 'vue-router';
import _ from 'underscore';
// import Scripts from '../Main/Scripts';

import service from './service';


export default {
    name: 'Search',

    data()  {
        return {
            detailModal: null,

            sortDropdown: null,
            searchKeyword: '',
            detailForm: {},

            loader: {
                searchResult: false,
                scripts: false,
            },

            page: {
                size: 5,
                current: 0,
                end: 1,
                list: [],
                listSize: 10,
            },

            documents: [],
            scripts: [],

            selectedDocument: { _index: '', _type: '', _score: 0, highlight: { doc: [] } },

            keyword: '',
            relationKeywords: [],

            visibleDetailSearch: false,
        };
    },

    methods: {
        onSearch(no = 0)  {
            if(no < 0)  no = 0;
            if(no > this.page.end)  no = this.page.end;

            if(this.searchKeyword && this.searchKeyword.length >= 1)   {
                this.$router.push(`/Search/${this.searchKeyword}/${no}`);
            } else    {
                this.showAlert({ message: '검색어를 입력해주세요.' });
            }
        },

        async getSearchResult() {
            if(!this.searchKeyword || isNaN(this.page.current))  return;

            this.loader.searchResult = true;

            // let searchResult = await service.getSearchResult(this.searchKeyword, this.page.current * this.page.listSize),
            let searchResult = await service.getAdvancedCombinationsResults(this.detailForm, this.page.current * this.page.listSize),
                pageStart = Math.floor(this.page.current / this.page.size) * this.page.size;

            this.page.end = Math.floor(searchResult.hitscount / 10);
            this.documents = searchResult.documents;
            this.keyword = searchResult.keyword;

            this.page.list = [];

            for(let no = pageStart; no <= this.page.end; no++)    {
                if(no >= pageStart + this.page.size)    break;
                this.page.list.push(no);
            }

            this.loader.searchResult = false;
        },

        async getScripts()  {
            this.loader.scripts = true;
            this.scripts = [];

            this.scripts = await service.getScripts(this.selectedDocument._index, this.selectedDocument._type);

            this.loader.scripts = false;
        },

        showScripts(document)   {
            this.selectedDocument = document;
            this.detailModal.show();
            this.getScripts();
        },

        onSortChange(value)    {

        },
    },

    watch: {
        $route: function(toRoute, fromRoute) {
            let params = toRoute.params;

            this.searchKeyword = params.value;
            this.page.current = Number(params.no);

            this.getSearchResult();
        },

        sortDropdown: function()    {
            this.sortDropdown.setSelected('score');
        },

        keyword: async function()   {
            this.relationKeywords = [];
            this.relationKeywords = await service.getRelationKeywords(this.searchKeyword);
        },

        visibleDetailSearch: function(value) {
            if(this.value === false)    this.onSearch(0);
        }
    },

    computed: {

    },

    directives: {

    },

    components: {
        Scripts,
    },

    beforeCreate()  {},
	created()	{},
    beforeMount()   {},
	mounted() 	{
        let params = this.$route.params;

        this.searchKeyword = params.value;
        this.page.current = Number(params.no);

        setTimeout(() => {
            this.getSearchResult();
        }, 0);
    },
    beforeUpdate()  {},
    updated()   {},
    beforeDestroy() {},
    destroyed() {}
};

</script>

<style scoped lang="less">
.main-wrap {
    margin:0;
}

input[type="text"]  {
    width: 400px;
}

.ui.list.link.page-wrap {
    a.item {
        &.active {
            color:#2185D0;
        }
    }
}

.doc-content {
    overflow: hidden;
    white-space: nowrap;
    text-overflow: ellipsis;

    b {
        background-color: #ffff00;
        color: #000000;
    }
}
</style>