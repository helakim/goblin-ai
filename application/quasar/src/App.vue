<template>
<section class="app-wrap">
    <div class="ui inverted borderless menu">
        <div class="item">
            <div class="ui tiny image">
                <img :src="require('./assets/logo.png')" alt="Goblin-Ai Qusare logo" />
            </div>
        </div>
        <div class="item header">
            <h3>{{title}}</h3>
        </div>
    </div>
    <router-view></router-view>
</section>
</template>

<script>
import Router from 'vue-router';
import jQuery from 'jquery';

import CONST from './const/const';

export default {
    name: 'app',

    props: {
        sidebar: {
            type: Object,
            default: null,
        }
    },

    data()  {
        return {
            title: '',
            menuItems: CONST.menuItems,
        };
    },

    components: {

    },

    methods: {
        onClickItem: function()   {
            this.sidebar.hide();
        },

        checkRouter()   {
            let routerPath = this.getRouterPath();

            this.changeTitle(routerPath);

            if(!routerPath.length)  {
                this.$el.style.height = `${this.getWindowHeight()}px`;
            } else {
                this.$el.style.height = `auto`;
            }
        },

        getRouterPath() {
            return (this.$route.matched[0] || {}).path || '';
        },

        changeTitle(routerPath)    {
            let title = '';

            this.$el.setAttribute('class', 'app-wrap');

            this.menuItems.some(item => {
                if(routerPath.indexOf(item.routerLink) >= 0) {
                    title = item.name;
                    this.$el.classList.add(item.icon);

                    return true;
                }
            });

            this.title = title;
        },
    },

    watch: {
        $route: function()  {
            this.checkRouter();
        }
    },

    beforeCreate()  {
        // if(this.$route.path === '/')    this.$router.push('Statist');
    },
	created()	{
        this.menuItems.some(item => {
            if(item.routerLink === this.$route.path)    {
                this.title = item.name;
                return true;
            }
        });
    },
    beforeMount()   {},
	mounted() 	{
        this.checkRouter();
    },
    beforeUpdate()  {},
    updated()   {},
    beforeDestroy() {},
    destroyed() {}
};

</script>

<style src="@/components/semantic/semantic.min.css"></style>
<style lang="less" src="./styles.less"></style>
<style lang="less" scoped>
.ui.inverted.borderless.menu {
    max-height:47px;
    margin-bottom:0;
}
</style>
