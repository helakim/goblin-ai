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
import more from 'highcharts/highcharts-more';
import _ from 'underscore';


more(Highcharts);

export default {
    name: 'StatistPieChart',

    props: [ 'isModal', 'loaderVisible', 'chartData' ],

    data()  {
        return {
            viewChartData: [],
            periodType: 'daily',
            chartContainerID: 'pie-container' + (this.isModal ? '-modal' : ''),
        }
    },

    methods: {
        onPeriodButtonClick(button, period)   {
            var activeButton = button.parentElement.querySelector('.blue');

            if(activeButton)    activeButton.classList.remove('blue');

            button.classList.add('blue');

            this.periodType = period;
        },

        drawChart() {
            Highcharts.chart(this.chartContainerID, {
                chart: {
                    type: 'pie',
                    marginTop: 0,
                    backgroundColor: 'transparent'
                },
                colors: this.$highChartsColors,
                title: {
                    text: '',
                    enabled: false,
                },
                subtitle: {
                    text: '',
                    enabled: false,
                },
                credits: {
                    enabled: false
                },
                plotOptions: {
                    series: {
                        dataLabels: {
                            enabled: true,
                            format: '{point.name}<br/>{point.y}건'
                        }
                    }
                },

                tooltip: {
                    headerFormat: '<span style="font-size:11px">{series.name}</span><br>',
                    pointFormat: '<span style="color:{point.color}">{point.name}</span>: <b>{point.y:.2f}%</b> of total<br/>',
                    enabled: false
                },

                series: [{
                    data: this.viewChartData
                }],
            });
        },
    },

    watch: {
        periodType: function(value)  {
            this.viewChartData = [];

            if(value)   {
                this.viewChartData = this.chartData[this.periodType].pie;
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
#pie-container {
    min-height:302px;

    &+div {
        text-align: center;
        margin-top:30%;
    }
}

#pie-container-modal {
    min-height:500px;

    &+div {
        text-align: center;
        margin:227px auto 226px;
    }
}
</style>

