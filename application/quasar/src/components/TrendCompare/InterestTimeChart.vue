<template>
<div class="ui fluid container">
    <div class="ui huge inverted header attached top">
        시간 흐름에 따른 관심도 변화
    </div>
    <div class="ui attached segment">
        <Loader :show="loaderVisible" />
        <div class="ui grid">
            <div class="three wide column">
                <div id="time-column-chart-container"></div>
            </div>
            <div class="thirteen wide column">
                <div id="time-line-chart-container"></div>
            </div>
        </div>
    </div>
</div>
</template>

<script>
import Highcharts from 'highcharts/highcharts';

const CURRENT_MONTH = (new Date).getMonth() + 1;
const MONTH_LIST = [
    CURRENT_MONTH - 3 + '월',
    CURRENT_MONTH - 2 + '월',
    CURRENT_MONTH - 1 + '월',
    CURRENT_MONTH + '월',
];

export default {
    name: 'TrendCompareResultInterestTimeChart',

    props: {
        loaderVisible: {
            type: Boolean,
            default: false,
        },

        queries: {
            type: Array,
            default: [],
        },

        trendFlow: {
            type: Object,
            default: {},
        },
    },

    data()  {
        return {

        };
    },

    watch: {
        trendFlow: function(value)   {
            let columnChartData = [],
                lineChartData = [];

            Object.keys(this.trendFlow).forEach(key => {
                let scores = this.trendFlow[key].map(row => row.doc_count);

                lineChartData.push({ name: key, data: scores, });
                columnChartData.push({ name: key, data: [ scores.reduce((a, b) => a + b) / scores.length ] });
            });

            this.drawLineChart({
                data: lineChartData,
            });

            this.drawColumnChart({
                categories: [ '평균' ],
                data: columnChartData,
            });
        },
    },

    computed: {},

    methods: {
        drawLineChart(value = {
            data: [],
        }) {
            Highcharts.chart('time-line-chart-container', {
                chart: {
                    type: 'line',
                    backgroundColor: 'transparent',
                    marginTop: 20,
                    height: 300,
                },
                colors: this.$semanticColors,
                title: {
                    text: ''
                },
                subtitle: {
                    text: ''
                },
                credits: {
                    enabled: false
                },
                xAxis: {
                    categories: MONTH_LIST
                },
                yAxis: {
                    title: {
                        enabled: false
                    }
                },
                legend: {
                    enabled: false,
                },
                plotOptions: {
                    line: {
                        dataLabels: {
                            enabled: false,
                            format: '{point.y}건'
                        },
                        marker: {
                            enabled: false
                        },
                    },
                },
                tooltip: {
                    shared: true,
                    useHTML: true,
                    backgroundColor: 'rgba(255,255,255, 1)',
                    borderWidth: 0,
                    style: {
                        color: '#232323'
                    },
                    headerFormat: '<strong>{point.key}</strong><table style="padding-top:1em">',
                    pointFormatter: function()  {
                        return `
                            <tr>
                                <td style="padding-right:2.5em;">${this.series.name}</td>
                                <td style="text-align: right"><b style="color: ${this.series.color}">${this.y} 건</b></td>
                            </tr>
                        `;
                    },
                    footerFormat: '</table>',
                },
                series: value.data
            });
        },

        drawColumnChart(value = {
            categories: [],
            data: [],
        }) {
            Highcharts.chart('time-column-chart-container', {
                chart: {
                    type: 'column',
                    backgroundColor: 'transparent',
                    marginTop: 20,
                    height: 300,
                },
                colors: this.$semanticColors,
                title: {
                    text: ''
                },
                subtitle: {
                    text: ''
                },
                credits: {
                    enabled: false
                },
                xAxis: {
                    categories: value.categories,
                },
                yAxis: {
                    title: {
                        enabled: false
                    },
                    visible: false,
                },
                legend: {
                    enabled: false,
                },
                tooltip: {
                    useHTML: true,
                    backgroundColor: 'rgba(255,255,255, 1)',
                    borderWidth: 0,
                    style: {
                        color: '#232323'
                    },
                    headerFormat: '<strong>{point.key}</strong><table style="padding-top:1em">',
                    pointFormatter: function()  {
                        return `
                            <tr>
                                <td style="padding-right:2.5em;">${this.series.name}</td>
                                <td style="text-align: right"><b style="color: ${this.series.color}">${this.y} 건</b></td>
                            </tr>
                        `;
                    },
                    footerFormat: '</table>',
                },
                series: value.data
            });
        },
    },

    components: {},

    directives: {},

    beforeCreate()  {},
    created()   {},
    beforeMount()   {},
    mounted()   {
        this.drawLineChart();
        this.drawColumnChart();
    },
    beforeUpdate()  {},
    updated()   {},
    beforeDestroy() {},
    destroyed() {},
}
</script>

