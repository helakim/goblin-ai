<template>
<div class="ui container basic segment no-padding">
    <table class="ui table">
        <thead>
        <tr class="active blue center aligned">
            <th colspan="4">
                <h3>{{title}}</h3>
            </th>
        </tr>
        <tr class="center aligned">
            <th>순위</th>
            <th>키워드</th>
            <th>점수</th>
            <!-- <th>변동폭</th> -->
        </tr>
        </thead>
        <tbody>
        <tr v-if="keywords.buckets && keywords.buckets.length"
            v-for="(item, index) in keywords.buckets"
            class="center aligned"
        >
            <td>{{index + 1}}</td>
            <td>{{item.key}}</td>
            <td>{{item.doc_count}}</td>
            <!-- <td>{{item.status || '-'}}</td> -->
        </tr>
        <tr v-if="!keywords.buckets || !keywords.buckets.length">
            <td class="center aligned" colspan="4">
                <div>
                    <i class="grey big warning circle icon"></i>
                    <p>조회 된 데이터가 없습니다</p>
                </div>
            </td>
        </tr>
        </tbody>
    </table>
    <Loader v-bind:show="loaderVisible" />
</div>
</template>

<script>
import _ from 'underscore';

export default {
    name: 'StatistHotKeyword',

    props: [ 'type', 'keywords', 'loaderVisible' ],

    data()  {
        return {

        }
    },

    methods: {

    },

    computed: {
        title: function()   {
            var title = '';

            switch(this.type)    {
                case 'DAILY': title = '일간'; break;
                case 'WEEK': title = '주간'; break;
                case 'MONTH': title = '월간'; break;
            }

            return `${title} 핫키워드`
        },
    },

    watch: {

    },

    directives: {

    },

    components: {

    },

    beforeCreate()  {},
	created()	{},
    beforeMount()   {},
	mounted() 	{},
    beforeUpdate()  {},
    updated()   {},
    beforeDestroy() {},
    destroyed() {}
}
</script>

<style scoped lang="less">
.ui.table {
    td[colspan] {
        height :42 * 10px;
    }
}
</style>

