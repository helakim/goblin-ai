<template>
<div class='ui container'>
    <div v-if="!isModal" class="ui blue border header">
        <h3>키워드 클라우드</h3>
    </div>
    <div class="ui container right aligned">
        <!-- <div class="ui buttons blue">
            <button class="ui button" type="button">일간</button>
            <button class="ui button" type="button">주간</button>
            <button class="ui button" type="button">월간</button>
        </div> -->

        <button
            v-if="!isModal"
            @click="$emit('update:expandView', 'show')"
            class="ui basic button icon">
            <i class="icon expand"></i>
        </button>
    </div>

    <div :id="keywordCloudID"
        class="ui middle centered aligned container grid basic segment no-padding">
        <div v-if="!words.length"
            class="ui grid container one column row">
            <div class="column center aligned">
                <div class="no-data-wrap">
                    <i class="grey big warning circle icon"></i>
                    <p>조회 된 데이터가 없습니다</p>
                </div>
            </div>
        </div>
        <Loader v-bind:show="loaderVisible" />
    </div>
</div>
</template>

<script>
import _ from 'underscore';
import cloud from 'd3-cloud';


const d3 = require('d3'),
    fill = d3.schemeCategory20.concat(d3.schemeCategory20b.concat(d3.schemeCategory20c));

export default {
    name: 'StatistCloud',

    props: {
        isModal: {
            type: Boolean,
            default: false,
        },
        loaderVisible: {
            type: Boolean,
            default: false,
        },
        words: {
            type: Array,
            default: [],
        }
    },

    data()  {
        return {
            keywordCloudID: 'keyword-cloud' + (this.isModal ? '-modal' : ''),
            cloud: null,
        }
    },

    methods: {
        drawCloud(words) {
            d3.select(`#${this.keywordCloudID}`).select('svg').remove();

            if(!words.length)   return;

            d3.select(`#${this.keywordCloudID}`)
                .append('svg')
                    .attr('width', this.cloud.size()[0])
                    .attr('height', this.cloud.size()[1])
                .append('g')
                    .attr('transform', 'translate(' + this.cloud.size()[0] / 2 + ',' + this.cloud.size()[1] / 2 + ')')
                .selectAll('text')
                    .data(words)
                .enter().append('text')
                    .style('font-size', function(d) { return d.size + 'px'; })
                    .style('font-family', 'sans-serif')
                    .style('fill', function(d, i) { return fill[i]; })
                    .attr('text-anchor', 'middle')
                    .attr('transform', function(d) {
                        return 'translate(' + [d.x, d.y] + ')rotate(' + d.rotate + ')';
                    })
                    .text(function(d) { return d.text; });
        }
    },

    directives: {

    },

    watch: {
        words: function()   {
            this.cloud.words(this.words.slice(0, 60));
            this.cloud.start();
        }
    },

    components: {

    },

    beforeCreate()  {},
	created()	{},
    beforeMount()   {},
	mounted() 	{
        this.cloud = cloud()
            .size(this.isModal ? [ 800, 640 ] : [ 300, 300 ])
            .words([])
            .padding(1)
            .rotate(function() { return 0; })
            .font('sans-serif')
            .fontSize(function(d) { return d.size; })
            .on('end', this.drawCloud);
    },
    beforeUpdate()  {},
    updated()   {},
    beforeDestroy() {},
    destroyed() {}
}
</script>

<style scoped lang='less'>
#keyword-cloud {
    height:300px !important;
}

#keyword-cloud-modal {
    height:640px !important;
}

.no-data-wrap {
    margin:121px auto;
}
</style>
