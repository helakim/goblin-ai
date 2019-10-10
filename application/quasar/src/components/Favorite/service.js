import Vue from 'vue';


class FavoriteKeyword {
    constructor(value = {
        buckets: [],
        doc_count_error_upper_bound: 0,
        sum_other_doc_count: 0,
    })   {
        this.buckets = value.buckets;
        this.doc_count_error_upper_bound = value.doc_count_error_upper_bound;
        this.sum_other_doc_count = value.sum_other_doc_count;
    };
};

export default {
    /**
     *
     * @param {Date} date
     * @returns {Map<string, FavoriteKeyword>}
     */
    getKeywordsForPeriod: async (date) => {
        return await Vue.http.get(`favorite/keyword/${date}`)
            .then(res => {
                let body = res.body || {},
                    lastMonth = new FavoriteKeyword(body.lastmonth.favorite_keyword_aggs),
                    lastWeek = new FavoriteKeyword(body.lastweek.favorite_keyword_aggs),
                    yesterday = new FavoriteKeyword(body.yesterday.favorite_keyword_aggs),
                    queryDay = new FavoriteKeyword(body.currentday.favorite_keyword_aggs);

                return new Map([
                    ['lastMonth', lastMonth],
                    ['lastWeek', lastWeek],
                    ['yesterday', yesterday],
                    ['queryDay', queryDay],
                ]);
            })
            .catch(error => error);
    },

    /**
     * @param {string} keyword
     * @returns {Map<string, FavoriteKeyword>}
     */
    getKeywordFlows: async (keyword) => {
        return await Vue.http.get(`favorite/keyword/flow/${keyword}`)
            .then(res => {
                let body = res.body,
                    p3Month = new FavoriteKeyword(body['3-d/month'].keyword_flow),
                    p2Month = new FavoriteKeyword(body['2-d/month'].keyword_flow),
                    p1Month = new FavoriteKeyword(body['1-d/month'].keyword_flow),
                    queryMonth = new FavoriteKeyword(body.month.keyword_flow);

                return new Map([
                    ['p3Month', p3Month],
                    ['p2Month', p2Month],
                    ['p1Month', p1Month],
                    ['queryMonth', queryMonth],
                ]);
            })
            .catch(error => error);
    },
};
