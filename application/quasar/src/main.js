// The Vue build version to load with the `import` command
// (runtime-only or standalone) has been set in webpack.base.conf with an alias.
import Vue from 'vue';
import Vuex from 'vuex';
import VueResource from 'vue-resource';
import jQuery from 'jquery';
import moment from 'moment';
import App from './App';
import router from './router';
import Vuemantic from '@/components/semantic/vuemantic';
import Background from '@/components/Background/Background';
import Sidebar from '@/components/Sidebar';
import Loader from '@/components/Loader';
import Alert from '@/components/Alert';
import DetailSearch from '@/components/DetailSearch/DetailSearch';


const APP_NAME = 'Goblin-Ai Qusare';

// Need to load jQuery global for Semantic-ui
Object.defineProperties(window, { jQuery: { get: () => jQuery }, $: { get: () => jQuery } });

Vue.config.productionTip = false;

Vue.use(Vuex);
Vue.use(VueResource);
Vue.use(Vuemantic);

if(!console.table)  {
    console.table = console.debug;
}

if(process.env.NODE_ENV !== 'development')    {
    console.debug = console.warn = console.table = function()   {};
}

Vue.http.interceptors.push((req, next) => {
    req.url = '/api/demo/' + req.url;

    console.debug(`------------------------------------ Request ------------------------------------`);
    console.debug(`[${APP_NAME}][Request][%s] url > %s`, moment(new Date).format('YYYY-MM-DD HH:mm:ss'), req.url);
    console.debug(`[${APP_NAME}][Request][%s] parmas >`, moment(new Date).format('YYYY-MM-DD HH:mm:ss'));
    console.table(req.parmas);

    setTimeout(() => {
        next(res => {
            var level = 'debug';

            switch(Math.floor(res.status / 100))  {
                case 4:
                case 5:
                    level = 'warn';
                    break;
            }

            console[level](`------------------------------------ Response ------------------------------------`);
            console[level](`[${APP_NAME}][Response][%s] status > %s`, moment(new Date).format('YYYY-MM-DD HH:mm:ss'), res.status);
            console[level](`[${APP_NAME}][Response][%s] body >`, moment(new Date).format('YYYY-MM-DD HH:mm:ss'));
            console.table(res.body);
        });
    }, 300);
});

Vue.component('Loader', Loader);
Vue.component('Background', Background);
Vue.component('DetailSearch', DetailSearch);
Vue.directive('focus', {
    inserted(el)  {
        el.focus();
    },
});
Vue.directive('fixed', {
    inserted(el)  {
        document.querySelector('body').insertBefore(el, null);
    },
    unbind(el) {
        el.remove();
    },
});
Vue.directive('toggle-element', {
    inserted(el, binding)  {
        let className = binding.value.className || 'active';

        el.classList.remove(className);

        el.addEventListener('click', function() {
            if (el.classList.contains(className))   {
                el.classList.remove(className);
            } else {
                el.classList.add(className);
            }
        });
    }
});

const VueSidebar = new Vue({
    el: '#app-sidebar',
    router,
    template: '<Sidebar :el="sidebar" @update:el="function(el) { sidebar = el; }" />',
    data: {
        sidebar: null,
    },
    components: { Sidebar }
});

const VueAlert = new Vue({
    el: '#app-alert',
    template: `
        <Alert
            :title="title"
            @update:title="function(value) { title = value; }"
            :message="message"
            @update:message="function(value) { message = value; }"
            :action="action"
            @update:action="function(value) { action = value; }"
        />
    `,
    data: {
        title: '',
        message: '',
        action: 'hide',
    },
    components: { Alert }
});

Vue.mixin({
    data()  {
        return {
            scrollTop: 0,
        };
    },
    created()   {
        this.$wordCloudColors = [ '#003f6a', '#08607e', '#1cb4e2', '#1374a8', '#49b0c1' ];
        this.$highChartsColors = [ '#4fb1bd', '#9bbb59', '#4bacc6', '#2c4d75', '#5f7530', '#276a7c', '#729aca', '#afc97a', '#6fbdd1', '#3a679c' ];
        this.$keywordColors = [ '#ff7c7c', '#ffeb7c', '#bbff7c', '#7cffca', '#7ccaff', '#b47cff', '#ff7ceb' ];
        this.$semanticColorsName = [ 'blue', 'red', 'orange', 'yellow', 'olive', 'green', 'teal', 'purple', 'violet', 'pink', 'brown', 'grey' ];
        this.$semanticColors = [ '#2185D0', '#db2828', '#f2711c', '#fbbd08', '#b5cc18', '#21ba45', '#00b5ad', '#a333c8', '#6435c9', '#e03997', '#a5673f', '#767676' ];

        document.addEventListener('scroll', (e) => {
            this.scrollTop = document.querySelector('html').scrollTop;
        });
    },
    methods: {
        showAlert({
            title = '',
            message = ''
        }) {
            VueAlert.title = title;
            VueAlert.message = message;
            VueAlert.action = 'show';
        },
        hideAlert() {
            VueAlert.action = 'hide';
        },

        getWindowWidth: () => window.innerWidth,
        getWindowHeight: () => window.innerHeight,
        getAppTopMenuHeight: () => $('.app-wrap > .ui.menu').height(),
    },
});

/* eslint-disable no-new */
const VueApp = new Vue({
    el: '#app',
    router,
    template: `
        <div>
            <Background />
            <App :sidebar="sidebar" />
            <Loader
                :show="visible"
                :inverted="false"
                :message="message" />
        </div>
    `,
    data()  {
        return {
            sidebar: VueSidebar.sidebar,

            visible: true,
            message: 'Loading'
        };
    },
    components: { Background, App },
    mounted()   {
        setInterval(() => this.message += '.', 500);
        setTimeout(() => this.visible = false, 2000);
    }
});
