import Vue from 'vue';


export default {
    test: () => {
        console.log('test');
    },
    getKeywords: async (date) => {
        return await Vue.http.get(`statistics/keyword/${date}`)
            .then(res => {
                var body = res.body;

                return {
                    daily: body.daily || { hot_keyword_weeks: { buckets: [] } },
                    week: body.week || { hot_keyword_weeks: { buckets: [] } },
                    month: body.month || { hot_keyword_month: { buckets: [] } },
                };
            })
            .catch(error => error);
    },
    getWordCloud: async (date) => {
        return await Vue.http.post(`statistics/wordcloud/`, {
                date: date
            })
            .then(res => res.body ? res.body.hot_keyword_list : { doc_count_error_upper_bound: 0, sum_other_doc_count: 0, buckets: [] })
            .then(hot_keyword_list => hot_keyword_list.buckets)
            .catch(error => error);
    },
    getChartDatas: async (date) => {
        return await Vue.http.post(`statistics/charts/`, {
                date: date
            })
            .then(res => res.body ? res.body.hot_keyword_list : { doc_count_error_upper_bound: 0, sum_other_doc_count: 0, buckets: [] })
            .catch(error => error);
    },
};
