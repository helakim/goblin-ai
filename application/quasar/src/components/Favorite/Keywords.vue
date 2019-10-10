<template>
<div class="ui container">
    <div class="ui blue border header">
        <h3>관심키워드</h3>
    </div>
    <div class="ui container basic segment no-padding">
        <table class="ui selectable table">
            <thead>
            <tr class="active blue center aligned">
                <th>순위</th>
                <th>키워드</th>
                <th>건수</th>
                <th>조회일</th>
                <th>전일</th>
                <th>전주</th>
                <th>전월</th>
            </tr>
            </thead>
            <tbody>
            <tr v-toggle-element="{className: 'active'}"
                v-if="tableData.length"
                v-for="(item, index) in tableData"
                @click="selectItem(item.key)"
                class="center aligned">
                <td v-show="false">
                    <input
                        v-model="active"
                        :id="item.key"
                        :value="item.key"
                        @click="e => e.stopPropagation()"
                        type="checkbox" />
                </td>
                <td>{{index + 1}}</td>
                <td>{{item.key}}</td>
                <td>{{item.total}}</td>
                <td>{{item.queryDay}}</td>
                <td>{{item.yesterday}}</td>
                <td>{{item.lastWeek}}</td>
                <td>{{item.lastMonth}}</td>
            </tr>
            <tr v-if="!tableData.length"
                class="center aligned">
                <td colspan="7">
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
</div>
</template>

<script>
import $ from 'jquery';

export default {
    name: 'FavoriteKeywords',

    props: [ 'type', 'tableData', 'loaderVisible' ],

    data()  {
        return {
            active: [],
        }
    },

    watch: {

    },

    methods: {
        selectItem(key)  {
            $(`#${key}`).trigger('click');

            this.$emit('update:selectKeyword', this.active);
        },
    },

    computed: {

    },

    watch: {
        tableData: function(value) {
            this.active = [];

            if(this.tableData.length >= 1)   {
                setTimeout(() => {
                    this.selectItem(this.tableData[0].key);
                    $('tr:first-child').addClass('active');
                }, 0);
            }
        },
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