<template>
<div class="ui container">
    <div class="ui blue border header">
        <h3>키워드별 추이</h3>
    </div>
    <div class="ui container basic segment no-padding">
        <div id="line-container"></div>
        <Loader v-bind:show="loaderVisible" />
    </div>
</div>
</template>

<script>
import Highcharts from 'highcharts/highcharts';
import _ from 'underscore';


export default {
    name: 'TrendLineChart',

    props: {
        loaderVisible: {
            type: Boolean,
            default: false,
        }
    },

    data()  {
        return {

        }
    },

    methods: {
        onPeriodButtonClick(button, period)   {
            var activeButton = button.parentElement.querySelector('.blue');

            if(activeButton)    activeButton.classList.remove('blue');

            button.classList.add('blue');

            this.drawChart(this.chartData[period])
        },

        drawChart(data = {
            categories: [],
            counts: [],
        }) {
            Highcharts.chart('line-container', {
                chart: {
                    type: 'line',
                    marginTop: 20,
                    height: 400
                },
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
                    categories: [ '일간', '주간', '월간' ]
                },
                yAxis: {
                    title: {
                        enabled: false
                    }
                },
                plotOptions: {
                    line: {
                        dataLabels: {
                            enabled: true
                        },
                        enableMouseTracking: false
                    }
                },
                series: [{
                    name: 'Tokyo',
                    data: [7.0, 6.9, 9.5,]
                }, {
                    name: 'London',
                    data: [3.9, 4.2, 5.7,]
                }]
            });
        },
    },

    watch: {
        chartData: function(chartData)   {
            this.onPeriodButtonClick(this.$el.querySelector('.ui.buttons button'), 'detail');
        }
    },

    directives: {

    },

    components: {

    },

    beforeCreate()  {},
	created()	{},
    beforeMount()   {},
	mounted() 	{

    },
    beforeUpdate()  {

    },
    updated()   {

    },
    beforeDestroy() {},
    destroyed() {}
}
</script>