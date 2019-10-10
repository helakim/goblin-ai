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
                            from: new Date
                        }"
                    ></Datepicker>
                </div>
                <button @click="onSearch" class="ui blue button">조회</button>
            </div>
        </div>
    </section>
    <section class="ui internally celled grid two column row">
        <div class="column">
            <Keywords
                :tableData="tableData"
                @update:selectKeyword="selectKeyword"
                :loaderVisible="loader.keywords" />
        </div>
        <div class="column">
            <LineChart
                :searchDate="searchDate"
                :chartData="chartData"
                :loaderVisible="loader.chart" />
        </div>
    </section>
</div>
</template>

<script>
import moment from 'moment';
import Datepicker from 'vuejs-datepicker';
import Highcharts from 'highcharts/highcharts';
import _ from 'underscore';

import service from './service';
import Keywords from './Keywords';
import LineChart from './LineChart';


class ChartData {
    constructor()   {
        this.key = '';
        this.monthly = {
            counts: [],
        };
        this.detail = {
            counts: [],
        };
    };
};


export default {
    name: 'Favorite',

    data()  {
        return {
            loader: {
                keywords: false,
                chart: false,
            },

            searchDate: new Date,
            keywords: new Map,
            tableData: [],
            chartData: new Map,
            selectedKeywords: new Map,
        };
    },

    methods: {
        async onSearch()  {
            this.loader.keywords = true;

            this.keywords = new Map;
            this.selectedKeywords = new Map;
            this.chartData = new Map;
            this.tableData = [];

            let date = moment(this.searchDate).format('YYYY-MM-DD'),
                keywordsForPeriod = await service.getKeywordsForPeriod(date);

            keywordsForPeriod.forEach((favoriteKeyword, period) => {
                favoriteKeyword.buckets.forEach(bucket => {
                    let key = bucket.key,
                        count = bucket.doc_count;

                    if(!this.keywords.has(key)) {
                        this.keywords.set(key, {
                            key: key,
                            total: 0,
                            lastMonth: 0,
                            lastWeek: 0,
                            yesterday: 0,
                            queryDay: 0,
                        });
                    }

                    this.keywords.get(key).total += count;
                    this.keywords.get(key)[period] = count || 0;
                });
            });

            this.keywords.forEach(keyword => {
                this.tableData.push(keyword);
            });

            this.tableData = this.tableData.sort((a, b) => b.total - a.total);

            this.loader.keywords = false;
        },

        async getChartData(keyword = {
            key: '',
            total: 0,
            lastMonth: 0,
            lastWeek: 0,
            yesterday: 0,
            queryDay: 0,
        })  {
            let keywordFlows = await service.getKeywordFlows(keyword.key),
                chartData = new ChartData;

            chartData.key = keyword.key;

            keywordFlows.forEach((favoriteKeyword, period) => {
                chartData.monthly.counts.push(favoriteKeyword.buckets[0].doc_count);
            });

            chartData.detail.counts = [
                keyword.yesterday + keyword.queryDay,
                keyword.lastWeek,
                keyword.lastMonth,
            ];

            return chartData;
        },

        async selectKeyword(keys = []) {
            let chartData = new Map(this.chartData);

            this.chartData = new Map;

            this.loader.chart = true;

            chartData.forEach((value, key) => {
                if(keys.indexOf(key) === -1)    chartData.delete(key);
            });

            for(let index = 0; index < keys.length; index++)    {
                let key = keys[index],
                    has = false;

                chartData.forEach((value, chartDataKey) => {
                    if(key === chartDataKey)    has = true;
                });

                if(!has)    {
                    chartData.set(key, await this.getChartData(this.keywords.get(key)));
                }
            }

            this.chartData = new Map(chartData);

            this.loader.chart = false;
        }
    },

    watch: {

    },

    computed: {

    },

    directives: {

    },

    components: {
        Datepicker,
        Keywords,
        LineChart,
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

.keyword-cloud-wrap {
    height:300px !important;
}
</style>
