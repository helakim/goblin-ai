<template>
<div class="ui container">
    <div class="ui blue border header">
        <h3>키워드 추이</h3>
    </div>
    <div class="ui container">
        <div class="ui blue label large icon comma-wrap">
            <i class="icon search"></i>
            키워드:
            <strong v-for="item in chartData.toJSON()" class="comma">
                {{item[0]}}
            </strong>
        </div>

        <div class="ui divider"></div>

        <div class="ui small buttons">
            <button
                @click="onPeriodButtonClick($event.target, CHART_TYPE.DETAIL)"
                class="ui button blue">
                {{searchDate.getMonth() + 1}}월 상세
            </button>
            <button
                @click="onPeriodButtonClick($event.target, CHART_TYPE.MONTHLY)"
                class="ui button">
                월별동향
            </button>
        </div>
    </div>
    <div class="ui container basic segment no-padding">
        <div id="line-container"></div>
        <Loader v-bind:show="loaderVisible" />
    </div>
</div>
</template>

<script>
import Highcharts from 'highcharts/highcharts';


const CHART_TYPE = {
    DETAIL: 'DETAIL',
    MONTHLY: 'MONTHLY',
};

const CHART_TYPE_CATEGORIES = {
    DETAIL: () => [ '일간', '주간', '월간' ],
    MONTHLY: (startMonth) => {
        return [ -3, -2, -1, 0 ].map((month) => {
            month = startMonth + month;
            month < 1 ? month += 12 : void(0);
            return month + '월';
        });
    }
};

export default {
    name: 'FavoriteLineChart',

    props: [ 'loaderVisible', 'searchDate', 'chartData' ],

    data()  {
        return {
            CHART_TYPE: CHART_TYPE,
            CHART_TYPE_CATEGORIES: CHART_TYPE_CATEGORIES,

            currentChartType: CHART_TYPE.DETAIL,
            datas: {
                detail: [],
                monthly: [],
            },
        }
    },

    methods: {
        onPeriodButtonClick(button, chartType)   {
            let activeButton = button.parentElement.querySelector('.blue');

            if(activeButton)    activeButton.classList.remove('blue');

            button.classList.add('blue');

            this.currentChartType = chartType;

            this.drawChart();
        },

        drawChart() {
            Highcharts.chart('line-container', {
                chart: {
                    type: 'line',
                    marginTop: 20,
                    height: 450
                },
                colors: this.$highChartsColors,
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
                    categories: this.CHART_TYPE_CATEGORIES[this.currentChartType](this.searchDate.getMonth() + 1)
                },
                yAxis: {
                    title: {
                        enabled: false
                    }
                },
                plotOptions: {
                    line: {
                        dataLabels: {
                            enabled: true,
                            format: '{point.y}건'
                        },
                        enableMouseTracking: false
                    }
                },
                series: this.datas[this.currentChartType.toLowerCase()],
            });
        },
    },

    watch: {
        chartData: function(value)    {
            this.datas.detail = [];
            this.datas.monthly = [];

            if(!this.chartData)    return;
            else {
                this.chartData.forEach(chartData => {
                    if(chartData)   {
                        this.datas.detail.push({
                            name: chartData.key,
                            data: chartData.detail.counts
                        });

                        this.datas.monthly.push({
                            name: chartData.key,
                            data: chartData.monthly.counts
                        });
                    }
                });
            }

            this.drawChart();
        },
    },

    directives: {

    },

    components: {

    },

    beforeCreate()  {},
	created()	{},
    beforeMount()   {},
	mounted() 	{
        this.drawChart(CHART_TYPE.DETAIL);
    },
    beforeUpdate()  {

    },
    updated()   {

    },
    beforeDestroy() {},
    destroyed() {}
}
</script>
