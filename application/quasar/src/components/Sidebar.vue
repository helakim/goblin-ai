<template>
<div v-show="sidebar"
    v-ui-sidebar="{
        el: 'sidebar',
        animation: 'push',
        setting: {
            duration: 0,
            dimPage: false,
            exclusive: true,
            closable: false,
            delaySetup: true,
        }
    }"
    class="ui icon sidebar inverted vertical compact menu">
    <a @click="toggleSidebarText"
        class="item">
        <div>
            <i class="icon large sidebar"></i>
            <span v-show="sidebarTextVisible">메뉴</span>
        </div>
    </a>
    <router-link v-if="sidebar"
        v-for="item in menuItems.slice(1)"
        :item="item"
        :key="item.name"
        :to="{ path: item.routerLink }"
        :active-class="'active'"
        class="item">
        <div>
            <i class="icon large material-icons">{{item.icon}}</i>
            <span v-show="sidebarTextVisible">
                {{item.name}}
            </span>
        </div>
    </router-link>
</div>
</template>

<script>
import CONST from '../const/const';

export default {
    name: 'Sidebar',

    props: {
        el: {
            type: Object,
            default: {},
        },
    },

    data()  {
        return {
            sidebar: null,
            menuItems: CONST.menuItems,
            sidebarTextVisible: false,
        };
    },

    watch: {
        $route: function()  {
            this.sidebarTextVisible = false;
        }
    },

    methods: {
        toggleSidebarText() {
            this.sidebarTextVisible ? this.sidebarTextVisible = false : this.sidebarTextVisible = true;
        },
    },

    mounted()   {
        this.sidebar.show();

        this.$el.addEventListener('resize', () => {
            this.sidebar.show();
        });

        this.$emit('update:el', this.sidebar);
    },
}
</script>

<style scoped lang="less">
@import "../const.less";

.ui.inverted.icon.menu {
    .item {
        text-align: left;
        // display: block;
        // padding: 0;

        // &.header {
        //     padding: 0.92857143em 1.14285714em;
        // }

        // span {
        //     display: block;
        //     padding: 0.92857143em 1.14285714em;
        // }

        & > div > i.icon {
            &.material-icons { vertical-align: -5px; }
            vertical-align: -2px;
        }

        &.active {
            &:before {
                width:4px;
                height:100%;
                background-color:@blue;
                content: '';
            }
        }
    }
}
</style>
