class DetailForm  {
    constructor()   {
        this.base = '';
        this.must = '';
        this.or = '';
        this.not = '';
    };

    static parse(str = '')  {
        let mustReg = /\"([^"]*)\"/,
            base = '',
            must = '',
            or = '',
            not = '',
            orIndexes = [],
            result = new DetailForm;

        if (mustReg.test(str))  must = mustReg.exec(str)[1];

        str = str.replace(mustReg, '').replace(/\"/gi, '').replace(/\s{2,}/, ' ').split(/\s/);

        orIndexes[str.indexOf('OR') - 1] = true;

        str.forEach((word, index) => {
            if (word.indexOf('-') === 0) {
                not += (word.replace('-', '') + ' ');
            } else {
                if (word.indexOf('OR') === 0) {
                    orIndexes[index - 1] = true;
                    orIndexes[index + 1] = true;
                } else {
                    let isOrIndex = orIndexes[index];

                    if (!isOrIndex) base += (word + ' ');
                    else or += (word + ' ');
                }
            }
        });

        orIndexes = [];

        result.base = base.trim();
        result.or = or.trim();
        result.not = not.trim();
        result.must = must;

        return result;
    };

    toJSON()    {
        return {
            base_keyword: this.base.length ? this.base.split(/\s/) : [],
            must_keyword: this.must.length ? this.must.split(/\s/) : [],
            or_keyword: this.or.length ? this.or.split(/\s/) : [],
            not_keyword: this.not.length ? this.not.split(/\s/) : [],
        };
    };

    toString()  {
        let str = this.base;

        if (this.or.length)    str += (' ' + this.or.split(/\s/).join(' OR '));
        if (this.must.length)  str += (' \"' + this.must + '\"');
        if (this.not.length)   str += (' -' + this.not.split(/\s/).join(' -'));

        return str;
    };
};

export { DetailForm };
