<template>
<div class="ui grid container main-wrap">
    <div v-if="type === 'tag'" class="column row">
        <iframe v-iframe src=""></iframe>
    </div>
    <div v-if="type === 'term'" class="column row">
        <iframe v-iframe src=""></iframe>
    </div>
</div>
</template>

<script>
export default {
    name: 'Cloud',

    data()  {
        return {
            type: '',
        };
    },

    watch: {
        $route: function()  {
            this.type = this.$route.params.type;
        }
    },

    directives: {
        iframe: {
            inserted(el)  {
                el.height = window.innerHeight - 70;

                window.addEventListener('resize', () => {
                    el.height = window.innerHeight - 70;
                });
            }
        }
    },

    beforeCreate()  {},
	created()	{
        this.type = this.$route.params.type;
    },
    beforeMount()   {},
	mounted() 	{},
    beforeUpdate()  {},
    updated()   {},
    beforeDestroy() {},
    destroyed() {}
}
</script>

<style scoped lang="less">
.ui.grid.container {
    margin-top:0;
    margin-bottom:0;
    width:100% !important;

    .column.row {
        padding-top:1.8rem;
        padding-bottom:0;

        iframe {
            display: block;
            width:100%;
            border:0;
        }
    }
}
</style>

