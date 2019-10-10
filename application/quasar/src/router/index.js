import Vue from 'vue';
import Router from 'vue-router';
import Main from '@/components/Main/Main';
import Statist from '@/components/Statist/Statist';
import KeywordTrend from '@/components/KeywordTrend/KeywordTrend';
import TrendCompare from '@/components/TrendCompare/TrendCompare';
import TrendCompareResult from '@/components/TrendCompare/Result';
import Favorite from '@/components/Favorite/Favorite';
import Search from '@/components/Search/Search';


Vue.use(Router);

export default new Router({
    routes: [
        {
            path: '/Main',
            name: 'Main',
            component: Main
        },
        {
            path: '/Statist',
            name: 'Statist',
            component: Statist
        },
        // {
        //     path: '/KeywordTrend',
        //     name: 'KeywordTrend',
        //     component: KeywordTrend
        // },
        {
            path: '/TrendCompare',
            name: 'TrendCompare',
            component: TrendCompare,
        },
        {
            path: '/TrendCompare/:query',
            name: 'TrendCompareResult',
            component: TrendCompareResult
        },
        {
            path: '/Favorite',
            name: 'Favorite',
            component: Favorite
        },
        {
            path: '/Search',
            name: 'Search',
            component: Search,
            children: [
                {
                    path: ':value/:no',
                    name: 'Search',
                    component: Search,
                }
            ],
        },
        // {
        //     path: '/Documents',
        //     name: 'Documents',
        //     component: Documents
        // },
        // {
        //     path: '/Cloud/:type',
        //     name: 'Cloud',
        //     component: Cloud
        // },
    ]
});
