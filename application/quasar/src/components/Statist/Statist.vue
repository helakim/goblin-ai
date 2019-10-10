<template>
<div class="ui fluid container main-wrap">
    <section class="ui internally celled grid one column row">
        <div class="column search-wrap">
            <div class="ui labeled action input">
                <div class="ui blue label">
                    날짜
                </div>
                <div class="ui icon input right aligned">
                    <i class="calendar icon"></i>
                    <Datepicker v-model="searchDate"
                        :format="'yyyy.MM.dd'"
                        :language="'ko'"
                        :disabled="{
                            from: disabledPicker
                        }"
                    ></Datepicker>
                </div>
                <button
                    @click="onSearch"
                    class="ui blue button">조회</button>
            </div>
        </div>
    </section>
    <section class="ui internally celled grid three column row">
        <div class="column">
            <HotKeywords
                :type="hotKeywordType.daily"
                :loaderVisible="loader.hotkeyWords"
                :keywords="keywords.daily.hot_keyword_weeks" />
        </div>
        <div class="column">
            <HotKeywords
                :type="hotKeywordType.week"
                :loaderVisible="loader.hotkeyWords"
                :keywords="keywords.week.hot_keyword_weeks" />
        </div>
        <div class="column">
            <HotKeywords
                :type="hotKeywordType.month"
                :loaderVisible="loader.hotkeyWords"
                :keywords="keywords.month.hot_keyword_month" />
        </div>
    </section>
    <section class="ui internally celled grid three column row">
        <div class="column">
            <Cloud
                :words="words"
                :loaderVisible="loader.wordCloud"
                @update:expandView="action => cloudModal[action]()" />
        </div>
        <div class="column" style="width:66.66666667%">
            <div class="ui blue border header">
                <h3>키워드 차트</h3>
            </div>
            <div class="ui grid two column">
                <div class="eight wide column">
                    <PieChart
                        :chartData="chartData"
                        :loaderVisible="loader.chart"
                        @update:expandView="action => pieChartModal[action]()" />
                </div>
                <div class="eight wide column">
                    <ColumnChart
                        :chartData="chartData"
                        :loaderVisible="loader.chart"
                        @update:expandView="action => columnChartModal[action]()" />
                </div>
            </div>
        </div>
    </section>

    <div v-ui-modal="{
            el: 'cloudModal',
            setting: {
                duration: 300,
            }
        }"
        class="ui modal">
        <i class="icon close"></i>
        <div class="header">
            키워드 클라우드
        </div>
        <div class="content">
            <Cloud
                :words="words"
                :loaderVisible="loader.wordCloud"
                :isModal="true" />
        </div>
    </div>
    <div v-ui-modal="{
            el: 'pieChartModal',
            setting: {
                duration: 300,
            }
        }"
        class="ui modal">
        <i class="icon close"></i>
        <div class="header">
            키워드 차트
        </div>
        <div class="content">
            <PieChart
                :chartData="chartData"
                :loaderVisible="loader.chart"
                :isModal="true" />
        </div>
    </div>
    <div v-ui-modal="{
            el: 'columnChartModal',
            setting: {
                duration: 300,
            }
        }"
        class="ui modal">
        <i class="icon close"></i>
        <div class="header">
            키워드 차트
        </div>
        <div class="content">
            <ColumnChart
                :chartData="chartData"
                :loaderVisible="loader.chart"
                :isModal="true" />
        </div>
    </div>
</div>
</template>

<script>
import moment from 'moment';
import Datepicker from 'vuejs-datepicker';

import HotKeywords from './HotKeywords';
import Cloud from './Cloud';
import PieChart from './PieChart';
import ColumnChart from './ColumnChart';

import service from './service';


const DISABLED_PICKER = new Date;

DISABLED_PICKER.setHours(23, 59, 59);

export default {
    name: 'Statist',

    data()  {
        return {
            cloudModal: null,
            pieChartModal: null,
            columnChartModal: null,

            loader: {
                hotkeyWords: false,
                wordCloud: false,
                chart: false,
            },

            disabledPicker: DISABLED_PICKER,
            searchDate: new Date,

            hotKeywordType: {
                daily: 'DAILY',
                week: 'WEEK',
                month: 'MONTH',
            },

            keywords: {
                daily: { hot_keyword_weeks: {} },
                week: { hot_keyword_weeks: {} },
                month: { hot_keyword_month: {} },
            },

            words: [],

            chartData: {
                daily: {
                    categories: [],
                    pie: [],
                    column: [],
                },
                week: {
                    categories: [],
                    pie: [],
                    column: [],
                },
                month: {
                    categories: [],
                    pie: [],
                    column: [],
                },
            },
        };
    },

    methods: {
        async onSearch()  {
            let date = moment(this.searchDate).format('YYYY-MM-DD');

            this.loader.hotkeyWords = true;
            this.loader.wordCloud = true;
            this.loader.chart = true;

            this.chartData = null;
            this.words = [];
            this.keywords = await service.getKeywords(date);
            this.processChartData(this.keywords);
            this.loader.hotkeyWords = false;
            this.loader.chart = false;

            let words = await service.getWordCloud(date),
                maxCount = 0;

            if(words.length)    {
                maxCount = words[0].doc_count;

                this.words = words.map(word => {
                    return {
                        text: word.key,
                        size: word.doc_count / maxCount * 100,
                    };
                });
            }

            this.loader.wordCloud = false;
        },

        processChartData(keywords)  {
            this.chartData = {
                daily: {
                    categories: [],
                    pie: [],
                    column: [],
                },
                week: {
                    categories: [],
                    pie: [],
                    column: [],
                },
                month: {
                    categories: [],
                    pie: [],
                    column: [],
                },
            };

            keywords.daily.hot_keyword_weeks.buckets.forEach(row => {
                this.chartData.daily.pie.push({
                    name: row.key,
                    y: row.doc_count,
                });
                this.chartData.daily.column.push(row.doc_count);
                this.chartData.daily.categories.push(row.key);
            });

            keywords.week.hot_keyword_weeks.buckets.forEach(row => {
                this.chartData.week.pie.push({
                    name: row.key,
                    y: row.doc_count,
                });
                this.chartData.week.column.push(row.doc_count);
                this.chartData.week.categories.push(row.key);
            });

            keywords.month.hot_keyword_month.buckets.forEach(row => {
                this.chartData.month.pie.push({
                    name: row.key,
                    y: row.doc_count,
                });
                this.chartData.month.column.push(row.doc_count);
                this.chartData.month.categories.push(row.key);
            });
        },
    },

    computed: {

    },

    directives: {

    },

    components: {
        Datepicker,
        HotKeywords,
        Cloud,
        PieChart,
        ColumnChart,
    },

    beforeCreate()  {},
	created()	{},
    beforeMount()   {},
	mounted() 	{
        this.onSearch();
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

.ui.input.right.aligned {
    input {
        text-align: right;
    }
}
</style>
