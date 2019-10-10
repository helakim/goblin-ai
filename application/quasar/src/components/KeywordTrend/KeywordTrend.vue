<template>
<div class="ui fluid container main-wrap">
    <section class="ui internally celled grid three column row">
        <div class="column">
            <Keywords
                :type="keywordType.daily"
                :keywords="[]"
                :loaderVisible="loader.keywords" />
        </div>
        <div class="column">
            <Keywords
                :type="keywordType.week"
                :keywords="[]"
                :loaderVisible="loader.keywords" />
        </div>
        <div class="column">
            <Keywords
                :type="keywordType.month"
                :keywords="[]"
                :loaderVisible="loader.keywords" />
        </div>
    </section>
    <section class="ui internally celled grid two column row">
        <div class="column">
            <BubbleChart
                :loaderVisible="loader.chart" />
        </div>
        <div class="column">
            <LineChart
                :loaderVisible="loader.chart" />
        </div>
    </section>
</div>
</template>

<script>
import Highcharts from 'highcharts/highcharts';
import more from 'highcharts/highcharts-more';
import _ from 'underscore';

import Keywords from './Keywords';
import LineChart from './LineChart';
import BubbleChart from './BubbleChart';


more(Highcharts);

export default {
    name: 'Trend',

    data()  {
        return {
            loader: {
                keywords: false,
                chart: false,
            },

            keywordType: {
                daily: 'DAILY',
                week: 'WEEK',
                month: 'MONTH',
            },

            keywords: {
                daily: [],
                week: [],
                month: [],
            }
        };
    },

    methods: {
        drawBubbleChart()   {

        },
        drawLineChart() {

        },
    },

    computed: {

    },

    directives: {

    },

    components: {
        Keywords,
        BubbleChart,
        LineChart,
    },

    beforeCreate()  {},
	created()	{},
    beforeMount()   {},
	mounted() 	{
        for(let i = 0; i < 10; i++) {
            this.keywords.daily.push({
                keyword: 'ABC',
                score: _.random(100, 2000),
                status: _.random(-10, 10)
            });
            this.keywords.week.push({
                keyword: 'ABC',
                score: _.random(100, 2000),
                status: _.random(-10, 10)
            });
            this.keywords.month.push({
                keyword: 'ABC',
                score: _.random(100, 2000),
                status: _.random(-10, 10)
            });
        }

        this.drawBubbleChart();
        this.drawLineChart();
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
</style>
