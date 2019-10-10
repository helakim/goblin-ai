<template>
<section class="ui grid container main-wrap">
    <div class="ten wide column row">
        <div class="ui grid container one column row">
            <Scripts />
            <Player />
        </div>
    </div>
    <div class="six wide column row">
        <div class="ui grid container one column row">
            <CheckResults />
            <JudgingResults />
        </div>
    </div>
</section>
</template>

<script>
export default {
    name: 'Main',

    directives: {

    },

    components: {
    },

    beforeCreate()  {},
	created()	{},
    beforeMount()   {},
	mounted() 	{

    },
    beforeUpdate()  {},
    updated()   {},
    beforeDestroy() {},
    destroyed() {}
};

</script>

<style scoped lang="less">
.main-wrap {
    margin:0;
}
</style>
