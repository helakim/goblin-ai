import $ from 'jquery';


class Search    {
    constructor(el, options = {
        /**
         * Settings for API call.
         */
        apiSettings: { action: 'search' },

        /**
         * Minimum characters to query for results
         */
        minCharacters: 1,

        /**
         * Whether search should show results on focus (must also match min character length)
         */
        searchOnFocus: true,

        /**
         * Named transition to use when animating menu in and out. Fade and slide down are available without including ui transitions
         */
        transition: 'fade',

        /**
         * Duration of animation events
         */
        duration: 300,

        /**
         * Maximum results to display when using local and simple search, maximum category count for category search
         */
        maxResults: 7,

        /**
         * Caches results locally to avoid requerying server
         */
        cache: true,

        /**
         * Specify a Javascript object which will be searched locally
         */
        source: false,

        /**
         * Whether the search should automatically select the first search result after searching
         */
        selectFirstResult: false,

        /**
         * Whether a "no results" message should be shown if no results are found. (These messages can be modified using the template object specified below)
         */
        showNoResults: false,

        /**
         * Return local results that match anywhere inside your content
         */
        searchFullText: true,

        /**
         * List mapping display content to JSON property, either with API or source.
         */
        fields: {},

        /**
         * List mapping display content to JSON property, either with API or source.
         */
        defaultFields: {
            categories      : 'results',     // array of categories (category view)
            categoryName    : 'name',        // name of category (category view)
            categoryResults : 'results',     // array of results (category view)
            description     : 'description', // result description
            image           : 'image',       // result image
            price           : 'price',       // result price
            results         : 'results',     // array of results (standard)
            title           : 'title',       // result title
            action          : 'action',      // "view more" object name
            actionText      : 'text',        // "view more" text
            actionURL       : 'url'          // "view more" url
        },

        /**
         * Specify object properties inside local source object which will be searched
         */
        searchFields: [ 'title', 'description' ],

        /**
         * Delay before hiding results after search blur
         */
        hideDelay: 0,

        /**
         * Delay before querying results on inputchange
         */
        searchDelay: 100,

        /**
         * Easing equation when using fallback Javascript animation
         */
        easing: 'easeIn',

        /**
         * Callback on element selection by user. The first parameter includes the filtered response results for that element. The function should return false to prevent default action (closing search results and selecting value).
         */
        onSelect: function(result, response)    {},

        /**
         * Callback after processing element template to add HTML to results. Function should return false to prevent default actions.
         */
        onResultsAdd: function(html)    {},

        /**
         * Callback on search query
         */
        onSearchQuery: function(query)  {},

        /**
         * Callback on server response
         */
        onResults: function(response)   {},

        /**
         * Callback when results are opened
         */
        onResultsOpen: function()   {},

        /**
         * Callback when results are closed
         */
        onResultsClose: function()  {},

        /**
         * Name used in log statements
         */
        name: 'Search',

        /**
         * Event namespace. Makes sure module teardown does not effect other events attached to an element.
         */
        namespace: 'search',

        /**
         * Regular expressions used for matching
         */
        regExp: {
            escape     : /[\-\[\]\/\{\}\(\)\*\+\?\.\\\^\$\|]/g,
            beginsWith : '(?:\s|^)'
        },

        /**
         * Selectors used to find parts of a module
         */
        selector: {
            prompt       : '.prompt',
            searchButton : '.search.button',
            results      : '.results',
            category     : '.category',
            result       : '.result'
        },

        /**
         * HTML5 metadata attributes used internally
         */
        metadata: {
            cache   : 'cache',
            results : 'results'
        },

        /**
         * Class names used to determine element state
         */
        className: {
            active  : 'active',
            empty   : 'empty',
            focus   : 'focus',
            loading : 'loading',
            pressed : 'down'
        },

        /**
         * Silences all console output including error messages, regardless of other debug settings.
         */
        silent: 'False',

        /**
         * Debug output to console
         */
        debug: false,

        /**
         * Show console.table output with performance metrics
         */
        performance: true,

        /**
         * Debug output includes all internal behaviors
         */
        verbose: false,

        /**
         * Selectors used to find parts of a module
         */
        error: {
            source      : 'Cannot search. No source used, and Semantic API module was not included',
            noResults   : 'Your search returned no results',
            logging     : 'Error in debug logging, exiting.',
            noTemplate  : 'A valid template name was not specified.',
            serverError : 'There was an issue with querying the server.',
            maxResults  : 'Results must be an array to use maxResults setting',
            method      : 'The method you called is not defined.'
        },
    })   {
        this.$el = $(el).search(options);
    };

    /**
     * Search for value currently set in search input
     */
    query(callback)	{ this._behavior('query', callback); };

    /**
     * Displays message in search results with text, using template matching type
     */
    displayMessage(text, type)	{ this._behavior('display message', text, type); };

    /**
     * Cancels current remote search query
     */
    cancelQuery()	{ this._behavior('cancel query'); };

    /**
     * Search local object for specified query and display results
     */
    searchLocal(query)	{ this._behavior('search local', query); };

    /**
     * Whether has minimum characters
     */
    hasMinimumCharacters()	{ return this._behavior('has minimum characters'); };

    /**
     * Search remote endpoint for specified query and display results
     */
    searchRemote(query, callback)	{ this._behavior('search remote', query, callback); };

    /**
     * Search object for specified query and return results
     */
    searchObject(query, object, searchFields)	{ return this._behavior('search object', query, object, searchFields); };

    /**
     * Whether search is currently focused
     */
    isFocused()	{ return this._behavior('is focused'); };

    /**
     * Whether search results are visible
     */
    isVisible()	{ return this._behavior('is visible'); };

    /**
     * Whether search results are empty
     */
    isEmpty()	{ return this._behavior('is empty'); };

    /**
     * Returns current search value
     */
    getValue()	{ return this._behavior('get value'); };

    /**
     * Returns JSON object matching searched title or id (see above)
     */
    getResult(value)	{ return this._behavior('get result', value); };

    /**
     * Sets search input to value
     */
    setValue(value)	{ this._behavior('set value', value); };

    /**
     * Reads cached results for query
     */
    readCache(query)	{ return this._behavior('read cache', query); };

    /**
     * Clears value from cache, if no parameter passed clears all cache
     */
    clearCache(query)	{ this._behavior('clear cache', query); };

    /**
     * Writes cached results for query
     */
    writeCache(query)	{ return this._behavior('write cache', query); };

    /**
     * Adds HTML to results and displays
     */
    addResults(html)	{ this._behavior('add results', html); };

    /**
     * Shows results container
     */
    showResults(callback)	{ this._behavior('show results', callback); };

    /**
     * Hides results container
     */
    hideResults(callback)	{ this._behavior('hide results', callback); };

    /**
     * Generates results using parser specified by settings.template
     */
    generateResults(response)	{ return this._behavior('generate results', response); };

    /**
     * Removes all events
     */
    destroy()	{ this._behavior('destroy'); };

    _behavior(...args) {
        return $.fn.search.apply(this.$el, args);
    };
};


export default {
    bind: function(el, binding) {

    },

    inserted: function(el, binding, vnode)    {
        var search = new Search(el, binding.value.setting);

        vnode.context[binding.value.el] = search;
    },

    update: function() {

    },

    componentUpdated: function() {

    },

    unbind: function(el) {

    },
};
