<template>
<div class="ui container">
    <div class="ui blue border header">
        <h3>급상승 키워드 차트</h3>
    </div>
    <div class="ui container right aligned">
        <div class="ui buttons blue">
            <button
                @click="onPeriodButtonClick($event.target, 'daily')"
                class="ui button">일간</button>
            <button
                @click="onPeriodButtonClick($event.target, 'week')"
                class="ui button">주간</button>
            <button
                @click="onPeriodButtonClick($event.target, 'month')"
                class="ui button">월간</button>
        </div>
    </div>
    <div class="ui container basic segment no-padding">
        <div id="bubble-container"></div>
        <Loader v-bind:show="loaderVisible" />
    </div>
</div>
</template>

<script>
import Highcharts from 'highcharts/highcharts';
import _ from 'underscore';


export default {
    name: 'TrendBubbleChart',

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
            Highcharts.chart('bubble-container', {
                chart: {
                    type: 'bubble',
                    plotBorderWidth: 1,
                    zoomType: 'xy',
                    marginTop: 30,
                    height: 400
                },

                legend: {
                    enabled: true
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
                    gridLineWidth: 1,
                    title: {
                        text: 'Daily fat intake'
                    },
                    labels: {
                        format: '{value} gr'
                    },

                },

                yAxis: {
                    startOnTick: false,
                    endOnTick: false,
                    title: {
                        enabled: false
                    },
                    labels: {
                        format: '{value} gr'
                    },
                    maxPadding: 0.2,
                },

                tooltip: {
                    useHTML: true,
                    headerFormat: '<table>',
                    pointFormat: '<tr><th colspan="2"><h3>{point.country}</h3></th></tr>' +
                        '<tr><th>Fat intake:</th><td>{point.x}g</td></tr>' +
                        '<tr><th>Sugar intake:</th><td>{point.y}g</td></tr>' +
                        '<tr><th>Obesity (adults):</th><td>{point.z}%</td></tr>',
                    footerFormat: '</table>',
                    followPointer: true
                },

                plotOptions: {
                    series: {
                        dataLabels: {
                            enabled: true,
                            format: '{point.name}'
                        }
                    }
                },

                series: [
                    {
                        data: [
                            { x: 95, y: 95, z: 13.8, name: 'BE', country: 'Belgium' },
                        ]
                    },
                    {
                        data: [
                            { x: 86.5, y: 102.9, z: 14.7, name: 'DE', country: 'Germany' }
                        ],
                    },
                    {
                        data: [
                            { x: 80.8, y: 91.5, z: 15.8, name: 'FI', country: 'Finland' }
                        ],
                    },
                    {
                        data: [
                            { x: 80.4, y: 102.5, z: 12, name: 'NL', country: 'Netherlands' }
                        ],
                    },
                    {
                        data: [
                            { x: 80.3, y: 86.1, z: 11.8, name: 'SE', country: 'Sweden' }
                        ],
                    },
                    {
                        data: [
                            { x: 78.4, y: 70.1, z: 16.6, name: 'ES', country: 'Spain' }
                        ],
                    },
                    {
                        data: [
                            { x: 74.2, y: 68.5, z: 14.5, name: 'FR', country: 'France' }
                        ],
                    },
                    {
                        data: [
                            { x: 73.5, y: 83.1, z: 10, name: 'NO', country: 'Norway' }
                        ],
                    },
                    {
                        data: [
                            { x: 71, y: 93.2, z: 24.7, name: 'UK', country: 'United Kingdom' }
                        ],
                    },
                    {
                        data: [
                            { x: 69.2, y: 57.6, z: 10.4, name: 'IT', country: 'Italy' }
                        ],
                    },
                    {
                        data: [
                            { x: 68.6, y: 20, z: 16, name: 'RU', country: 'Russia' }
                        ],
                    },
                    {
                        data: [
                            { x: 65.5, y: 126.4, z: 35.3, name: 'US', country: 'United States' }
                        ],
                    },
                    {
                        data: [
                            { x: 65.4, y: 50.8, z: 28.5, name: 'HU', country: 'Hungary' }
                        ],
                    },
                    {
                        data: [
                            { x: 63.4, y: 51.8, z: 15.4, name: 'PT', country: 'Portugal' }
                        ],
                    },
                    {
                        data: [
                            { x: 64, y: 82.9, z: 31.3, name: 'NZ', country: 'New Zealand' }
                        ]
                    },
                ]
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

