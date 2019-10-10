import Vue from 'vue';

export default {
    getTrendCompare: async (value) => {
        return await Vue.http.get(`trendcompare/${encodeURIComponent(value)}`)
            .then(res => res.body)
            .catch(error => error);
    },
};
