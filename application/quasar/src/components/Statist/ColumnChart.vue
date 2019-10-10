<template>
<div class="ui container">
    <div class="ui container right aligned">
        <div class="ui buttons">
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

        <button
            v-if="!isModal"
            @click="$emit('update:expandView', 'show')"
            class="ui basic button icon">
            <i class="icon expand"></i>
        </button>
    </div>
    <div class="ui container basic segment no-padding">
        <div v-show="viewChartData.length" :id="chartContainerID"></div>
        <div v-if="!viewChartData.length">
            <i class="grey big warning circle icon"></i>
            <p>조회 된 데이터가 없습니다</p>
        </div>
        <Loader v-bind:show="loaderVisible" />
    </div>
</div>
</template>

<script>
import Highcharts from 'highcharts/highcharts';
import _ from 'underscore';


export default {
    name: 'StatistColumnChart',

    props: [ 'isModal', 'loaderVisible', 'chartData' ],

    data()  {
        return {
            viewChartData: [],
            viewChartCategories: [],
            periodType: 'daily',
            chartContainerID: 'column-container' + (this.isModal ? '-modal' : ''),
        }
    },

    methods: {
        onPeriodButtonClick(button, period)   {
            var activeButton = button.parentElement.querySelector('.blue');

            if(activeButton)    activeButton.classList.remove('blue');

            button.classList.add('blue');

            this.periodType = period;
        },

        drawChart()    {
            Highcharts.chart(this.chartContainerID, {
                chart: {
                    type: 'column',
                    marginTop: 20,
                    backgroundColor: 'transparent'
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
                    categories: this.viewChartCategories,
                    crosshair: true
                },
                yAxis: {
                    min: 0,
                    title: {
                        enabled: false
                    }
                },
                tooltip: {
                    headerFormat: '<span style="font-size:10px">{point.key}</span><table>',
                    pointFormat: '<tr><td style="color:{series.color};padding:0">{series.name}: </td>' +
                        '<td style="padding:0"><b>{point.y:.1f} mm</b></td></tr>',
                    footerFormat: '</table>',
                    shared: true,
                    useHTML: true,
                    enabled: false
                },
                plotOptions: {
                    column: {
                        pointPadding: 0.2,
                        borderWidth: 0,
                    }
                },
                legend: {
                    enabled: false
                },
                series: [
                    {
                        name: '',
                        data: this.viewChartData
                    }
                ]
            });
        },
    },

    watch: {
        periodType: function(value)  {
            this.viewChartData = [];
            this.viewChartCategories = [];

            if(value)   {
                this.viewChartData = this.chartData[this.periodType].column;
                this.viewChartCategories = this.chartData[this.periodType].categories;
            }

            this.drawChart();
        },

        chartData: function(value)   {
            if(value == null)  {
                this.periodType = null;
            } else  {
                this.onPeriodButtonClick(this.$el.querySelector('.ui.buttons button'), 'daily');
            }
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

<style scoped lang="less">
#column-container {
    min-height:302px;

    &+div {
        text-align: center;
        margin-top:30%;
    }
}

#column-container-modal {
    min-height:500px;

    &+div {
        text-align: center;
        margin:227px auto 226px;
    }
}
</style>
