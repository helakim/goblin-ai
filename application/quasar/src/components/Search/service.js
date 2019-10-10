import Vue from 'vue';
import { DetailForm } from '../DetailSearch/model';


class Document {
    constructor(document = {
        _id: '',
        _index: '',
        _score: 0,
        _source: { doc: '', _doc: '', timestamp: '', file_name: '' },
        _type: '',
        highlight: { doc: [] },
    })   {
        this._id = document._id;
        this._index = document._index;
        this._score = document._score;
        this._source = document._source;
        this._type = document._type;
        this.highlight = document.highlight;
    };
};

class SearchResult {
    /**
     * @param {Object} searchResult
     * @param {number} searchResult.hitscount
     * @param {Array<Document>} searchResult.document
     */
    constructor(searchResult = {
        hitscount: 0,
        document: [],
    })   {
        this.hitscount = searchResult.hitscount;
        this.documents = searchResult.document;
        this.keyword = '';
    };
};

class Keyword {
    /**
     * @param {Object} keyword
     * @param {string} keyword.key
     * @param {number} keyword.doc_count
     */
    constructor(keyword = {
        key: '',
        doc_count: 0,
    })   {
        this.key = keyword.key;
        this.doc_count = keyword.doc_count;
    };
};

export default {
    test: () => {
        console.log('test');
    },

    /**
     * @param {string} value
     * @param {number} no
     */
    getSearchResult: async (value, no = 0) => {
        return await Vue.http.get(`document/search/${encodeURIComponent(value)}/${no}`)
            .then(res => {
                let searchResult = new SearchResult(res.body);

                searchResult.keyword = value;

                return searchResult;
            })
            .catch(error => error);
    },

    /**
     * @param {string} value
     */
    getRelationKeywords: async (value) => {
        return await Vue.http.get(`document/keyword/relation/${encodeURIComponent(value)}`)
            .then(res => {
                if(!Array.isArray(res.body[0])) return [];

                let keywords = [];

                res.body[0].forEach(keyword => {
                    if(keyword.key !== value)   keywords.push(new Keyword(keyword));
                });

                return keywords;
            })
            .catch(error => error);
    },

    /**
     * @param {string} index
     * @param {string} type
     */
    getScripts: async (index, type) => {
        return await Vue.http.get(`document/details/${index}/${type}`)
            .then(res => {
                let scripts = res.body;

                return scripts;
            })
            .catch(error => error);
    },

    /**
     * @param {DetailForm} detailForm
     * @param {number} no
     */
    getAdvancedCombinationsResults: async (detailForm, no = 0) => {
        let body = Object.assign(detailForm);

        body._cidx = no;

        return await Vue.http.post(`advanced/combinations`, body)
            .then(res => {
                let searchResult = new SearchResult(res.body);

                searchResult.keyword = detailForm.base_keyword[0];

                return searchResult;
            })
            .catch(error => error);
    },
};
